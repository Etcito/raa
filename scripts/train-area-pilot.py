#!/usr/bin/env python3
"""Fit the FP32 area pilot to an immutable set of capture arrays."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ABI = Path(__file__).resolve().parents[1] / "crates/raa-graph/src/exec_ext/compute/learned"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_f32(root, record, count):
    data = (root / record["file"]).read_bytes()
    if len(data) != 4 * count or sha(data) != record["sha256"]:
        raise ValueError(f"capture length or hash mismatch: {record['file']}")
    values = torch.tensor(struct.unpack(f"<{count}f", data), dtype=torch.float32)
    if not values.isfinite().all():
        raise ValueError(f"non-finite capture values: {record['file']}")
    return values


def gather_pairs(raw, previous):
    neighbors = F.unfold(F.pad(raw[:, :2], (1, 1, 1, 1), mode="replicate"), 3)
    neighbors = neighbors.reshape(raw.shape[0], 2, 9, *raw.shape[2:])
    return torch.cat((neighbors, previous.unsqueeze(2)), dim=2)


def pair_ratios(pairs):
    weight = pairs[:, 1]
    supported = weight > 0
    ratios = pairs[:, 0] / torch.where(supported, weight, 1.0)
    return ratios, weight, supported


def stable_visibility(logits, pairs):
    ratios, weight, supported = pair_ratios(pairs)
    # Cancel the common (1 - history) factor from numerator and denominator.
    # The remaining previous/current multiplier is h/(1-h) = exp(gate_logit).
    log_mix = torch.cat((logits[:, :9].log_softmax(dim=1), logits[:, 9:10]), dim=1)
    log_mass = log_mix + torch.where(supported, weight, 1.0).log()
    log_mass = log_mass.masked_fill(~supported, -torch.inf)
    nonempty = supported.any(dim=1, keepdim=True)
    # Avoid an all-minus-infinity softmax on the zero-denominator branch.
    normalized = torch.where(nonempty, log_mass, 0.0).softmax(dim=1)
    visibility = (normalized * ratios).sum(dim=1)
    return torch.where(nonempty.squeeze(1), visibility, 1.0)


def oracle(raw, ref):
    previous = torch.where(raw[:, 10:11] == 1, raw[:, 2:4], 0.0)
    ratios, _, supported = pair_ratios(gather_pairs(raw, previous))
    nonempty = supported.any(dim=1)
    low = torch.where(nonempty, ratios.masked_fill(~supported, torch.inf).amin(dim=1), 1.0)
    high = torch.where(nonempty, ratios.masked_fill(~supported, -torch.inf).amax(dim=1), 1.0)
    # Every nonnegative mixture lies in this interval; independent per-pixel taps
    # may choose any value in it, including limits of the softmax/sigmoid.
    error = ref.clamp(min=low, max=high) - ref
    return dict(oracleMse=error.square().mean().item(),
                outsideOracleIntervalPixels=(error != 0).sum().item(),
                emptySupportPixels=(~nonempty).sum().item())


def renderer_layer(layer, x):
    # The shader walks taps/channels from the bias with one FP32 FMA per term.
    # Accumulate in FP64 momentarily, then round to FP32 at each fused operation.
    def fma(a, b, c):
        return (a.double() + b.double() * c.double()).float()

    weight = layer.weight
    result = layer.bias.view(1, -1, 1, 1)
    if layer.kernel_size == (1, 1):
        for channel in range(weight.shape[1]):
            result = fma(result, weight[:, channel, 0, 0].view(1, -1, 1, 1),
                         x[:, channel:channel + 1])
    else:
        height, width = x.shape[-2:]
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        for y in range(3):
            for col in range(3):
                result = fma(result, weight[:, 0, y, col].view(1, -1, 1, 1),
                             x[:, :, y:y + height, col:col + width])
    return result


class Pilot(nn.Module):
    def __init__(self, manifest, blob):
        super().__init__()
        self.layers = nn.ModuleList()
        self.scales = torch.tensor(manifest["normalizationScales"]).view(1, 12, 1, 1)
        for weight, bias in zip(manifest["tensors"][::2], manifest["tensors"][1::2]):
            shape = weight["shape"]
            depthwise = len(shape) == 3
            channels = shape[0] if depthwise else shape[1]
            layer = nn.Conv2d(channels, shape[0], 3 if depthwise else 1,
                              padding=1 if depthwise else 0, padding_mode="replicate",
                              groups=channels if depthwise else 1)
            with torch.no_grad():
                for param, spec in ((layer.weight, weight), (layer.bias, bias)):
                    values = struct.unpack_from(f"<{param.numel()}f", blob, spec["offset"])
                    param.copy_(torch.tensor(values).reshape(param.shape))
            self.layers.append(layer)

    def forward(self, raw, *, training_visibility=False):
        valid = raw[:, 10:11] == 1
        previous = torch.where(valid, raw[:, 2:4], 0.0)
        x = torch.cat((raw[:, :2], previous, raw[:, 4:]), dim=1) * self.scales
        for layer in self.layers[:-1]:
            x = (layer(x) if training_visibility else renderer_layer(layer, x)).relu()
        head = self.layers[-1]
        logits = head(x) if training_visibility else renderer_layer(head, x)
        visibility = stable_visibility(logits, gather_pairs(raw, previous))
        if training_visibility:
            return visibility
        spatial = logits[:, :9].softmax(dim=1)
        history = torch.where(valid, logits[:, 9:10].sigmoid(), 0.0)
        return torch.cat((spatial, history, logits[:, 10:11].sigmoid(), visibility.unsqueeze(1)), dim=1)

    def export(self, manifest):
        blob = bytearray()
        for layer, specs in zip(self.layers, zip(manifest["tensors"][::2], manifest["tensors"][1::2])):
            for param, spec in zip((layer.weight, layer.bias), specs):
                values = param.detach().flatten().tolist()
                if not all(map(math.isfinite, values)):
                    raise ValueError("non-finite fitted weights")
                blob.extend(bytes(spec["offset"] - len(blob)))
                data = struct.pack(f"<{len(values)}f", *values)
                blob.extend(data)
                spec["sha256"] = sha(data)
        manifest["weightsSha256"] = sha(blob)
        manifest.pop("packageSha256")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["packageSha256"] = sha(canonical + blob + (ABI / "golden.json").read_bytes())
        return blob


def train(capture, output, steps, learning_rate):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    capture_bytes = capture.read_bytes()
    dataset = json.loads(capture_bytes)
    if dataset["format"] != "raa.area-pilot-capture/1" or not dataset["frames"]:
        raise ValueError("expected a nonempty area pilot capture manifest")
    frames = []
    seen = set()
    for frame in dataset["frames"]:
        identity = frame["id"]
        seeds = frame["referenceSeeds"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("frame IDs must be nonempty and unique")
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError(f"invalid reference seeds for {identity}")
        width, height = frame["size"]
        if any(type(d) is not int or d <= 0 for d in (width, height)):
            raise ValueError(f"invalid extent for {identity}")
        raw = read_f32(capture.parent, frame["input"], width * height * 12)
        raw = raw.reshape(1, height, width, 12).permute(0, 3, 1, 2)
        ref = read_f32(capture.parent, frame["reference"], width * height)
        frames.append((identity, seeds, raw, ref.reshape(1, height, width)))

    manifest = json.loads((ABI / "manifest.json").read_bytes())
    initial_hash = manifest["weightsSha256"]
    model = Pilot(manifest, (ABI / "weights.fp32le").read_bytes())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def inference_losses():
        losses = [(model(raw)[:, 11] - ref).square().mean().item() for _, _, raw, ref in frames]
        if not all(map(math.isfinite, losses)):
            raise ValueError("non-finite FP32 inference loss")
        return losses

    initial_losses = inference_losses()
    initial = sum(initial_losses) / len(frames)
    best_loss, best_step = initial, 0
    best_weights = {name: value.clone() for name, value in model.state_dict().items()}
    max_grad_norm, clipped_steps = 0.0, 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for identity, _, raw, ref in frames:
            loss = (model(raw, training_visibility=True) - ref).square().mean() / len(frames)
            if not loss.isfinite():
                raise ValueError(f"non-finite training loss at step {step}, frame {identity}")
            loss.backward()
        try:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item()
        except RuntimeError as error:
            raise ValueError(f"non-finite gradient at step {step}") from error
        max_grad_norm = max(max_grad_norm, grad_norm)
        clipped_steps += grad_norm > 1.0
        optimizer.step()
        if step % 25 == 0 or step == steps:
            score = sum(inference_losses()) / len(frames)
            if score < best_loss:
                best_loss, best_step = score, step
                best_weights = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_weights)

    checks = []
    errors = []
    with torch.no_grad():
        for (identity, seeds, raw, ref), initial_mse in zip(frames, initial_losses):
            result = model(raw)
            delta = result[:, 11] - ref
            width, height = raw.shape[3], raw.shape[2]
            checks.append(dict(name=identity, size=[width, height], origin=[0, 0],
                               extent=[width, height],
                               input=raw.permute(0, 2, 3, 1).flatten().tolist(),
                               output=result.permute(0, 2, 3, 1).flatten().tolist()))
            bound = oracle(raw, ref)
            mse = delta.square().mean().item()
            if mse < bound["oracleMse"] - 1e-6:
                raise ValueError(f"renderer-equivalent fit escaped the mixture bound on {identity}")
            errors.append(dict(id=identity, referenceSeeds=seeds, initialMse=initial_mse, mse=mse,
                               maxAbsoluteError=delta.abs().max().item(), **bound,
                               gapMse=mse - bound["oracleMse"]))
    blob = model.export(manifest)
    report = dict(captureSha256=sha(capture_bytes), initializationSha256=initial_hash,
                  steps=steps, learningRate=learning_rate, torchVersion=torch.__version__,
                  initialLoss=initial, finalLoss=best_loss, bestStep=best_step,
                  oracleLoss=sum(e["oracleMse"] for e in errors) / len(errors),
                  maxGradientNorm=max_grad_norm, clippedSteps=clipped_steps, frames=errors)
    files = {"manifest.json": json.dumps(manifest, indent=2) + "\n",
             "trainer-vectors.json": json.dumps(checks, allow_nan=False) + "\n",
             "training-report.json": json.dumps(report, indent=2, allow_nan=False) + "\n"}
    output.mkdir()
    (output / "weights.fp32le").write_bytes(blob)
    for name, data in files.items():
        (output / name).write_text(data, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--learning-rate", default=0.003, type=float)
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("steps and learning rate must be positive and finite")
    train(args.capture, args.output, args.steps, args.learning_rate)
