#!/usr/bin/env python3
"""Overfit one optional lighting head on fixed FP32 captures using CPU PyTorch."""

import argparse
import math
import struct
from pathlib import Path

import torch
from torch import nn

from optional_head_capture import CHANNELS, POLICY, json_bytes, read_capture, sha

SEEDS = {"indirect-diffuse": 701, "indirect-specular": 702, "ao-visibility": 703}
HIDDEN = 8


def unpack_frame(frame, raw, target, inputs, outputs):
    width, height = frame["size"]
    raw = torch.tensor(raw, dtype=torch.float32, device="cpu").reshape(height, width, len(inputs))
    target = torch.tensor(target, dtype=torch.float32, device="cpu").reshape(height, width, len(outputs))
    source, confidence, known, fresh = raw[..., -len(POLICY):].unbind(dim=-1)
    depth = raw[..., inputs.index("metricDepth")]
    foreground = depth > 0
    current = raw[..., :len(outputs)]
    ao = len(outputs) == 1
    neutral = 1.0 if ao else 0.0

    for name, mask in (("sourceValid", source), ("confidenceValid", known), ("freshlyTraced", fresh)):
        if ((mask != 0) & (mask != 1)).any():
            raise ValueError(f"{frame['id']}: {name} must be binary")
    if (depth < 0).any() or not foreground.any():
        raise ValueError(f"{frame['id']}: expected foreground at positive metric depth")
    if (source[~foreground] != 0).any() or (known > source).any() or (fresh > source).any():
        raise ValueError(f"{frame['id']}: inconsistent source validity masks")
    if ((confidence < 0) | (confidence > 1) | ((known == 0) & (confidence != 0))).any():
        raise ValueError(f"{frame['id']}: invalid source confidence")
    if (current[source == 0] != neutral).any() or (target[~foreground] != neutral).any():
        raise ValueError(f"{frame['id']}: expected neutral values at invalid source or background pixels")
    for name, values in (("input", current), ("reference", target)):
        if (values < 0).any() or (ao and (values > 1).any()):
            raise ValueError(f"{frame['id']}: {name} values outside the signal range")
    normals = raw[..., inputs.index("normalX"):inputs.index("metricDepth")]
    if (normals.abs() > 1).any():
        raise ValueError(f"{frame['id']}: invalid normal components")
    if "roughness" in inputs:
        roughness = raw[..., inputs.index("roughness")]
        if ((roughness < 0) | (roughness > 1)).any():
            raise ValueError(f"{frame['id']}: invalid roughness")
    return raw, target, foreground


class Head(nn.Module):
    def __init__(self, inputs, outputs, seed):
        super().__init__()
        self.ao = len(outputs) == 1
        scales = [0.125 if name == "metricDepth" else 1.0 for name in inputs]
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32, device="cpu"))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.hidden = nn.Linear(len(inputs), HIDDEN, dtype=torch.float32, device="cpu")
            self.output = nn.Linear(HIDDEN, len(outputs), dtype=torch.float32, device="cpu")

    def forward(self, raw):
        logits = self.output(self.hidden(raw * self.scales).relu())
        return logits.sigmoid() if self.ao else nn.functional.softplus(logits)

    def weights(self):
        blob = bytearray()
        tensors = []
        for name, param in self.named_parameters():
            values = param.detach().flatten().tolist()
            if not all(map(math.isfinite, values)):
                raise ValueError("non-finite fitted weights")
            data = struct.pack(f"<{len(values)}f", *values)
            tensors.append(dict(name=name, offset=len(blob), shape=list(param.shape), sha256=sha(data)))
            blob.extend(data)
        return blob, tensors


def mse(delta):
    return delta.square().mean().item()


def score(model, frames):
    with torch.no_grad():
        losses = [mse(model(raw[mask]) - target[mask]) for _, raw, target, mask in frames]
    if not all(map(math.isfinite, losses)):
        raise ValueError("non-finite FP32 inference loss")
    return losses


def frame_metrics(raw, target, fitted, foreground):
    fitted_delta = fitted - target
    source_delta = raw[..., :target.shape[-1]] - target
    source, _, known, fresh = (raw[..., -len(POLICY):] == 1).unbind(dim=-1)

    def metrics(mask):
        count = mask.sum().item()
        result = dict(pixels=count)
        if count:
            result.update(inputMse=mse(source_delta[mask]), mse=mse(fitted_delta[mask]),
                          maxAbsoluteError=fitted_delta[mask].abs().max().item())
        return result

    return dict(**metrics(foreground), slices={
        "freshlyTraced": metrics(foreground & fresh),
        "validReused": metrics(foreground & source & ~fresh),
        "missingSource": metrics(foreground & ~source),
        "validConfidence": metrics(foreground & known),
        "missingConfidence": metrics(foreground & ~known),
    })


def train(capture, output, signal, steps, learning_rate):
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive and finite")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs, outputs = CHANNELS[signal]
    source, capture_hash, records = read_capture(capture, signal)
    frames = [(frame, *unpack_frame(frame, raw, target, inputs, outputs)) for frame, raw, target in records]
    model = Head(inputs, outputs, SEEDS[signal])
    initial_weights, _ = model.weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_losses = score(model, frames)
    initial_loss = sum(initial_losses) / len(frames)
    best_loss, best_step = initial_loss, 0
    best_weights = {name: param.clone() for name, param in model.state_dict().items()}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for frame, raw, target, mask in frames:
            loss = (model(raw[mask]) - target[mask]).square().mean() / len(frames)
            if not loss.isfinite():
                raise ValueError(f"non-finite loss at step {step} on {frame['id']}")
            loss.backward()
        try:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        except RuntimeError as error:
            raise ValueError(f"non-finite gradient at step {step}") from error
        optimizer.step()
        if step % 25 == 0 or step == steps:
            loss = sum(score(model, frames)) / len(frames)
            if loss < best_loss:
                best_loss, best_step = loss, step
                best_weights = {name: param.clone() for name, param in model.state_dict().items()}
    model.load_state_dict(best_weights)

    vectors, reports = [], []
    with torch.no_grad():
        for (frame, raw, target, mask), initial_mse in zip(frames, initial_losses):
            fitted = torch.full_like(target, 1.0 if model.ao else 0.0)
            fitted[mask] = model(raw[mask])
            record = frame["signals"][signal]
            reports.append(dict(
                id=frame["id"], viewId=frame["viewId"], frameId=frame["frameId"], timeMs=frame["timeMs"],
                size=frame["size"], inputSeeds=record["inputSeeds"],
                referenceMetadata=record["referenceMetadata"], initialMse=initial_mse,
                **frame_metrics(raw, target, fitted, mask),
            ))
            vectors.append(dict(id=frame["id"], size=frame["size"], input=raw.flatten().tolist(),
                                output=fitted.flatten().tolist()))

    weights, specs = model.weights()
    manifest = dict(
        format="raa.optional-head-cpu-fit/1", signal=signal, captureSource=source, captureSha256=capture_hash,
        inputChannels=inputs, outputChannels=outputs, layout="HWC", dtype="f32le",
        normalizationScales=model.scales.tolist(), hiddenChannels=HIDDEN, hiddenActivation="relu",
        outputActivation="sigmoid" if model.ao else "softplus", backgroundValue=1.0 if model.ao else 0.0,
        weights=dict(file="weights.fp32le", byteLength=len(weights), sha256=sha(weights), tensors=specs),
    )
    report = dict(
        signal=signal, captureSource=source, captureSha256=capture_hash, device="cpu",
        torchVersion=torch.__version__, initializationSeed=SEEDS[signal], initializationSha256=sha(initial_weights),
        optimizer="Adam", steps=steps, learningRate=learning_rate, bestStep=best_step,
        inputLoss=sum(frame["inputMse"] for frame in reports) / len(reports),
        initialLoss=initial_loss, finalLoss=best_loss, frames=reports,
    )
    files = {
        "weights.fp32le": weights,
        "trainer-vectors.json": json_bytes(vectors),
        "training-report.json": json_bytes(report),
        "manifest.json": json_bytes(manifest),
    }
    output.mkdir()
    for name, data in files.items():
        (output / name).write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="optional head capture manifest")
    parser.add_argument("output", type=Path, help="new directory for the fitted weights and report")
    parser.add_argument("--signal", required=True, choices=CHANNELS)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    args = parser.parse_args()
    try:
        train(args.capture, args.output, args.signal, args.steps, args.learning_rate)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.exit(1, f"{error}\n")
