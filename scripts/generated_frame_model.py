"""Bounded FP32 refinement of the causal target-time warp on CPU."""

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import torch
from torch import nn
from torch.nn import functional as F

from real_frame_model import configure_cpu, grid, json_bytes, sha, tensor_json


FORMAT = "raa.generated-frame-model/1"
FEATURES = [
    "logWarpR", "logWarpG", "logWarpB", "logFreshR", "logFreshG", "logFreshB",
    "warpValid", "freshValid", "targetDepthZ", "motionU", "motionV",
    "lightingChanged", "sourceAgeOver8", "motionReject",
]
LAYOUT = (
    ("stem", (12, 14), 0, 672),
    ("spatial", (12, 3, 3), 720, 1152),
    ("mix", (12, 12), 1200, 1776),
    ("head", (3, 12), 1824, 1968),
)
WEIGHTS_BYTES = 1984


@dataclass
class Packet:
    warp: torch.Tensor
    fresh: torch.Tensor
    depth: torch.Tensor
    motion: torch.Tensor
    lighting: torch.Tensor
    age: int


class GeneratedFrame(nn.Module):
    """The learned residual never overwrites trusted repair or admits missing pixels."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(14, 12, 1)
        self.spatial = nn.Conv2d(12, 12, 3, groups=12)
        self.mix = nn.Conv2d(12, 12, 1)
        self.head = nn.Conv2d(12, 3, 1)

    def forward(self, packet):
        warp_valid = packet.warp[:, 3:4] > 0.5
        fresh_valid = packet.fresh[:, 3:4] > 0.5
        warp = torch.where(warp_valid, packet.warp[:, :3].clamp_min(0), 0.0)
        fresh = torch.where(fresh_valid, packet.fresh[:, :3].clamp_min(0), 0.0)
        age = torch.full_like(packet.depth, min(max(packet.age, 0), 8) / 8)
        feature = torch.cat((warp.log1p(), fresh.log1p(), warp_valid, fresh_valid, packet.depth,
                             packet.motion[:, :2], packet.lighting, age,
                             packet.motion[:, 2:3]), dim=1)
        stem = self.stem(feature).relu()
        encoded = (stem + self.mix(self.spatial(F.pad(stem, (1, 1, 1, 1), mode="replicate")))).relu()
        log_residual = 0.25 * self.head(encoded).tanh()
        refined = (warp.log1p() + log_residual).expm1().clamp(0, 65504)
        learnable = warp_valid & ~fresh_valid & (packet.lighting < 0.5)
        output = torch.where(fresh_valid, fresh, torch.where(learnable, refined, warp))
        return {
            "feature": feature, "encoded": encoded, "logResidual": log_residual, "output": output,
            "coverage": warp_valid | fresh_valid, "learnable": learnable,
        }


def analytic(packet):
    available = packet.warp[:, 3:4] > 0.5
    base = torch.where(available, packet.warp[:, :3], 0.0)
    return torch.where(packet.fresh[:, 3:4] > 0.5, packet.fresh[:, :3], base).clamp_min(0)


def seeded(seed):
    configure_cpu()
    torch.manual_seed(seed)
    model = GeneratedFrame()
    # Start from the analytic result while keeping trainable features in the trunk.
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    return model


def tensor_records():
    for name, shape, weight, bias in LAYOUT:
        for suffix, dimensions, offset in (("weight", shape, weight), ("bias", (shape[0],), bias)):
            count = 1
            for length in dimensions:
                count *= length
            yield {"name": f"{name}.{suffix}", "shape": list(dimensions), "dtype": "f32le",
                   "offset": offset, "byteLength": count * 4}


def export_weights(model):
    blob = bytearray(WEIGHTS_BYTES)
    state = model.state_dict()
    for record in tensor_records():
        start = record["offset"]
        count = record["byteLength"] // 4
        values = state[record["name"]].detach().flatten().tolist()
        struct.pack_into(f"<{count}f", blob, start, *values)
    return bytes(blob)


def load_weights(manifest, blob):
    if (manifest["format"] != FORMAT or manifest["modelId"] != "generated-frame-v1"
            or manifest["tensors"] != list(tensor_records())
            or manifest["featureChannels"] != FEATURES or len(blob) != WEIGHTS_BYTES
            or manifest["weightsSha256"] != sha(blob)):
        raise ValueError("incompatible generated-frame weights")
    model = GeneratedFrame()
    state = model.state_dict()
    end = 0
    for record in tensor_records():
        start = record["offset"]
        if any(blob[end:start]):
            raise ValueError("generated-frame weights must use zero padding")
        count = record["byteLength"] // 4
        end = start + record["byteLength"]
        tensor = torch.tensor(struct.unpack_from(f"<{count}f", blob, start))
        if not tensor.isfinite().all():
            raise ValueError("generated-frame weights must be finite FP32")
        state[record["name"]] = tensor.reshape(state[record["name"]].shape)
    if any(blob[end:]):
        raise ValueError("generated-frame weights must use zero padding")
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def golden_vectors(model):
    size = (2, 3)
    x, y = grid(size)
    warp = torch.stack((x + 0.1, y + 0.2, x * y * 300 + 0.15, torch.ones_like(x)))[None]
    warp[:, :, 0, 0] = 0
    warp[:, :, 0, 1] = 0
    fresh = torch.zeros_like(warp)
    fresh[:, :, 0, 0] = torch.tensor([0.6, 0.4, 0.3, 1.0])
    depth = (0.2 + x / 8)[None, None]
    motion = torch.stack((x * 0.05, y * -0.03, x * 0.0, depth[0, 0]))[None]
    lighting = torch.zeros_like(depth)
    lighting[:, :, 0, 0] = 1
    packet = Packet(warp, fresh, depth, motion, lighting, 3)
    result = model(packet)
    inputs = dict(warp=warp, fresh=fresh, targetDepth=depth, motion=motion, lightingChanged=lighting)
    return {"format": "raa.generated-frame-golden/1", "layout": "HWC", "f32Absolute": 1e-5,
            "f32Relative": 1e-5, "vectors": [{
                "id": "odd-repair-and-invalid", "sourceAgeRealFrames": packet.age,
                "inputs": {k: tensor_json(v) for k, v in inputs.items()},
                "outputs": {k: tensor_json(v) for k, v in result.items()},
            }]}


def write_package(output: Path, model, *, kind, seed, provenance=None):
    weights = export_weights(model)
    golden_data = json_bytes(golden_vectors(model))
    manifest = {
        "format": FORMAT, "modelId": "generated-frame-v1", "kind": kind, "seed": seed,
        "weights": "weights.fp32le", "weightsSha256": sha(weights), "weightsByteLength": len(weights),
        "golden": "golden.json", "goldenSha256": sha(golden_data), "tensorLayout": "NHWC",
        "precision": "fp32", "tensorAlignment": 16, "tensors": list(tensor_records()),
        "featureChannels": FEATURES, "cadence": "admitted predictive presentation only",
        "colorDomain": "nonnegative scene-linear HDR at fixed exposure 1",
        "trainedHeads": ["frame-generation"] if kind == "windows-capture-fit" else [],
        "shippingQualification": "unverified",
    }
    if provenance is not None:
        manifest["provenance"] = provenance
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["packageSha256"] = sha(canonical + weights + golden_data)
    reloaded = load_weights(manifest, weights)
    if json_bytes(golden_vectors(reloaded)) != golden_data:
        raise ValueError("generated-frame package changed CPU golden vectors")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    (output / "weights.fp32le").write_bytes(weights)
    (output / "golden.json").write_bytes(golden_data)
    return manifest, reloaded
