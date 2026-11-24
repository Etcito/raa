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

    def forward(self, raw):
        valid = raw[:, 10:11] == 1
        previous = torch.where(valid, raw[:, 2:4], 0.0)
        x = torch.cat((raw[:, :2], previous, raw[:, 4:]), dim=1) * self.scales
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        logits = self.layers[-1](x)
        spatial = logits[:, :9].softmax(dim=1)
        history = torch.where(valid, logits[:, 9:10].sigmoid(), 0.0)
        neighbors = F.unfold(F.pad(raw[:, :2], (1, 1, 1, 1), mode="replicate"), 3)
        neighbors = neighbors.reshape(raw.shape[0], 2, 9, *raw.shape[2:])
        current = (neighbors * spatial.unsqueeze(1)).sum(dim=2)
        pair = torch.lerp(current, previous, history)
        denominator = pair[:, 1:2]
        positive = denominator > 0
        ratio = pair[:, :1] / torch.where(positive, denominator, 1.0)
        visibility = torch.where(positive, ratio, 1.0)
        return torch.cat((spatial, history, logits[:, 10:11].sigmoid(), visibility), dim=1)

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

    def loss(raw, ref):
        return (model(raw)[:, 11] - ref).square().mean()

    with torch.no_grad():
        initial = sum(loss(raw, ref).item() for _, _, raw, ref in frames) / len(frames)
    for _ in range(steps):
        optimizer.zero_grad()
        for _, _, raw, ref in frames:
            (loss(raw, ref) / len(frames)).backward()
        optimizer.step()

    checks = []
    errors = []
    with torch.no_grad():
        for identity, seeds, raw, ref in frames:
            result = model(raw)
            delta = result[:, 11] - ref
            width, height = raw.shape[3], raw.shape[2]
            checks.append(dict(name=identity, size=[width, height], origin=[0, 0],
                               extent=[width, height],
                               input=raw.permute(0, 2, 3, 1).flatten().tolist(),
                               output=result.permute(0, 2, 3, 1).flatten().tolist()))
            errors.append(dict(id=identity, referenceSeeds=seeds, mse=delta.square().mean().item(),
                               maxAbsoluteError=delta.abs().max().item()))
    blob = model.export(manifest)
    report = dict(captureSha256=sha(capture_bytes), initializationSha256=initial_hash,
                  steps=steps, learningRate=learning_rate, torchVersion=torch.__version__,
                  initialLoss=initial, finalLoss=sum(e["mse"] for e in errors) / len(errors),
                  frames=errors)
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
