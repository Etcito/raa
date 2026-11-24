"""Offline FP32 reference and portable weight ABI for the real-frame model."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "raa.real-frame-model/1"
GUIDES = [
    "normalX", "normalY", "normalZ", "logDepthOver8", "roughness",
    "motionU", "motionV", "historyValid", "reactive", "jitterX", "jitterY", "geometryValid",
]
SPECS = (
    ("stem", (12, 12), 0, 576),
    ("spatial", (12, 3, 3), 624, 1056),
    ("mix", (12, 12), 1104, 1680),
    ("area", (9, 26), 1728, 2672),
    ("recurrent", (24, 33), 2720, 5888),
    ("reflection", (3, 12), 5984, 6128),
    ("sr", (8, 12), 6144, 6528),
)
WEIGHTS_BYTES = 6560
PARITY_ABS = 1e-5
PARITY_REL = 1e-5


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def grid(size):
    height, width = size
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                          torch.arange(width, dtype=torch.float32), indexing="ij")
    return x, y


def at(image, x, y):
    """Integer pixel gather with active-rectangle clamp."""
    height, width = image.shape[-2:]
    return image[:, :, y.long().clamp(0, height - 1), x.long().clamp(0, width - 1)]


def taps(x, y):
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    return (
        (x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
        (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy),
    )


def sample_bilinear(image, x, y):
    return sum(at(image, tx, ty) * weight for tx, ty, weight in taps(x, y))


def display_coordinates(source, output, jitter):
    height, width = source
    out_height, out_width = output
    x, y = grid(output)
    sx, sy = width / out_width, height / out_height
    return (x + 0.5) * sx - 0.5 + jitter[0], (y + 0.5) * sy - 0.5 + jitter[1], sx, sy


@dataclass
class Frame:
    hdr: torch.Tensor
    normal_roughness: torch.Tensor
    depth: torch.Tensor
    motion: torch.Tensor
    reactive: torch.Tensor
    history_valid: torch.Tensor
    jitter: tuple
    output_size: tuple
    reset: bool = False
    area_pairs: torch.Tensor | None = None
    reflection: torch.Tensor | None = None


@dataclass
class History:
    state: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    jitter: tuple
    output_size: tuple


def valid_geometry(depth, normal):
    return (depth > 0) & (normal.square().sum(dim=1, keepdim=True) > 0.25)


class RealFrame(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(12, 12, 1)
        self.spatial = nn.Conv2d(12, 12, 3, groups=12)
        self.mix = nn.Conv2d(12, 12, 1)
        self.area = nn.Conv2d(26, 9, 1)
        self.recurrent = nn.Conv2d(33, 24, 1)
        self.reflection = nn.Conv2d(12, 3, 1)
        self.sr = nn.Conv2d(12, 8, 1)

    def encode(self, frame, previous):
        depth = frame.depth
        normal = frame.normal_roughness[:, :3]
        size = depth.shape[-2:]
        geometry_valid = valid_geometry(depth, normal)
        normal = torch.where(geometry_valid, normal, 0.0)
        reactive = frame.reactive.clamp(0, 1)
        valid = torch.zeros_like(geometry_valid)
        warped = depth.new_zeros((1, 12, *size))
        if (previous is not None and not frame.reset and previous.depth.shape == depth.shape
                and previous.output_size == frame.output_size):
            height, width = size
            x, y = grid(size)
            px = x - frame.motion[0, 0] * width + previous.jitter[0] - frame.jitter[0]
            py = y - frame.motion[0, 1] * height + previous.jitter[1] - frame.jitter[1]
            inside = (px >= -0.5) & (px < width - 0.5) & (py >= -0.5) & (py < height - 0.5)
            nearest_x, nearest_y = (px + 0.5).floor(), (py + 0.5).floor()
            old_depth = at(previous.depth, nearest_x, nearest_y)
            old_normal = at(previous.normal, nearest_x, nearest_y)
            depth_ok = (old_depth - depth).abs() <= (0.025 * depth).clamp_min(0.02)
            normal_ok = (old_normal * normal).sum(1, keepdim=True) >= 0.8
            valid = (inside & geometry_valid & (old_depth > 0) & depth_ok & normal_ok
                     & (frame.history_valid > 0.5) & (frame.motion[:, 2:3] < 0.5))
            warped = sample_bilinear(previous.state, px, py)
        gate = valid * (1 - reactive)
        jitter = [torch.full_like(depth, value) for value in frame.jitter]
        guide = torch.cat((normal, depth.clamp_min(0).log1p() / 8,
                           frame.normal_roughness[:, 3:4], frame.motion[:, :2], valid,
                           reactive, *jitter, geometry_valid), dim=1)
        stem = self.stem(guide).relu()
        encoded = (stem + self.mix(self.spatial(F.pad(stem, (1, 1, 1, 1), mode="replicate")))).relu()
        return encoded, warped, gate, valid

    def reconstruct_area(self, encoded, history, pairs):
        height, width = pairs.shape[-2:]
        s, u = pairs[:, :1], pairs[:, 1:2]
        center = visibility_ratio(s, u)
        logits = self.area(torch.cat((encoded, history, center, u.log1p()), dim=1))
        weights = logits.softmax(dim=1).reshape(1, 1, 9, height, width)
        samples = F.unfold(F.pad(pairs, (1, 1, 1, 1), mode="replicate"), 3)
        return (samples.reshape(1, 2, 9, height, width) * weights).sum(dim=2)

    def visibility(self, encoded, history, pairs):
        total = self.reconstruct_area(encoded, history, pairs)
        return visibility_ratio(total[:, :1], total[:, 1:2])

    def forward(self, frame, previous=None):
        encoded, warped, gate, valid = self.encode(frame, previous)
        gated = warped * gate
        refl = frame.reflection
        if refl is None:
            refl = frame.hdr.new_zeros((1, 6, *frame.hdr.shape[-2:]))
        reflect_valid = refl[:, 3:4] > 0.5
        rgb = torch.where(reflect_valid, refl[:, :3], 0.0)
        late = torch.cat((encoded, gated, frame.hdr.log1p(), rgb.log1p(), reflect_valid,
                          torch.where(reflect_valid, refl[:, 4:5] / 4, 0.0),
                          torch.where(reflect_valid, refl[:, 5:6].clamp_max(32) / 32, 0.0)), dim=1)
        candidate, retain = self.recurrent(late).chunk(2, dim=1)
        keep = gate * retain.sigmoid()
        state = candidate.tanh() * (1 - keep) + warped * keep
        coefficients = self.sr(state)
        reflection = torch.where(reflect_valid, residual(rgb, self.reflection(state), 0.5), 0.0)
        output = {
            "encoded": encoded, "warped": gated, "historyGate": gate, "historyValid": valid,
            "state": state, "coefficients": coefficients, "reflection": reflection,
            "display": decode_display(frame.hdr, coefficients, frame.output_size, frame.jitter),
        }
        if frame.area_pairs is not None:
            area = self.reconstruct_area(encoded, gated, frame.area_pairs)
            output["areaPairs"] = area
            output["areaVisibility"] = visibility_ratio(area[:, :1], area[:, 1:2])
        normal = frame.normal_roughness[:, :3]
        geometry = valid_geometry(frame.depth, normal)
        return output, History(state, torch.where(geometry, frame.depth, 0.0),
                               torch.where(geometry, normal, 0.0), frame.jitter, frame.output_size)


def visibility_ratio(s, u):
    supported = u > 1e-8
    return torch.where(supported, (s / torch.where(supported, u, 1.0)).clamp(0, 1), 1.0)


def residual(rgb, logits, amplitude):
    return (rgb.log1p() + amplitude * logits.tanh()).expm1().clamp_min(0)


def decode_display(hdr, coefficients, size, jitter):
    x, y, sx, sy = display_coordinates(hdr.shape[-2:], size, jitter)
    source = torch.cat((hdr, coefficients), dim=1)
    rgb_sum = 0.0
    residual_sum = 0.0
    weight_sum = 0.0
    for tx, ty, baseline in taps(x, y):
        values = at(source, tx, ty)
        c0, c1, c2, c3, c4 = values[:, 3:8].split(1, dim=1)
        bias = c0 + c1 * (x - tx) + c2 * (y - ty) + c3 * sx + c4 * sy
        weight = baseline * (0.5 * bias.tanh()).exp()
        rgb_sum = rgb_sum + weight * values[:, :3]
        residual_sum = residual_sum + weight * values[:, 8:11]
        weight_sum = weight_sum + weight
    return residual(rgb_sum / weight_sum, residual_sum / weight_sum, 0.25)


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def seeded(seed):
    configure_cpu()
    torch.manual_seed(seed)
    return RealFrame()


def tensor_records():
    for name, shape, weight, bias in SPECS:
        for suffix, dimensions, offset in (("weight", shape, weight), ("bias", (shape[0],), bias)):
            count = 1
            for length in dimensions:
                count *= length
            yield {"name": f"{name}.{suffix}", "shape": list(dimensions), "dtype": "f32le",
                   "offset": offset, "byteLength": count * 4}


def export_weights(model):
    blob = bytearray(WEIGHTS_BYTES)
    state = model.state_dict()
    for spec in tensor_records():
        values = state[spec["name"]].detach().flatten().tolist()
        start, size = spec["offset"], spec["byteLength"]
        blob[start:start + size] = struct.pack(f"<{len(values)}f", *values)
    return bytes(blob)


def load_weights(manifest, blob):
    if (manifest["format"] != FORMAT or manifest["tensors"] != list(tensor_records())
            or len(blob) != WEIGHTS_BYTES or manifest["weightsSha256"] != sha(blob)):
        raise ValueError("incompatible real-frame weights")
    model = RealFrame()
    weights = model.state_dict()
    end = 0
    for spec in tensor_records():
        start, size = spec["offset"], spec["byteLength"]
        if any(blob[end:start]):
            raise ValueError("real-frame weight padding must be zero")
        end = start + size
        values = torch.tensor(struct.unpack(f"<{size // 4}f", blob[start:end]))
        if not values.isfinite().all():
            raise ValueError("nonfinite real-frame weights")
        weights[spec["name"]] = values.reshape(weights[spec["name"]].shape)
    model.load_state_dict(weights)
    return model.eval()


def tensor_json(image):
    hwc = image.squeeze(0).permute(1, 2, 0).contiguous()
    return {"shape": list(hwc.shape), "data": hwc.flatten().float().tolist()}


@torch.no_grad()
def golden_vectors(model):
    """Three sequential odd-extent cases. Seeded data is ABI evidence only."""
    previous = None
    vectors = []
    for step in range(3):
        size = (2, 3)
        x, y = grid(size)
        hdr = torch.stack((x + 0.3 * step, y + 0.2, x * y * 20 + 0.1))[None]
        depth = (torch.ones((1, 1, *size)) * 3)
        normal = torch.zeros((1, 4, *size))
        normal[:, 2:3] = 1
        normal[:, 3:4] = x / 4
        motion = torch.zeros((1, 3, *size))
        reactive = torch.zeros_like(depth)
        history_valid = torch.ones_like(depth)
        jitter = (0.125 * step, -0.25 * step)
        if step == 1:
            motion[:, 0] = 1 / 6
            motion[:, 2, 0, 1] = 1
            depth[:, :, 0, 2] = 5
            reactive[:, :, 1, 0] = 1
            reactive[:, :, 1, 1] = 0.35
        if step == 2:
            history_valid[:, :, 0, 0] = 0
            normal[:, 2, 1, 1] = -1
        area = torch.stack((x * y * 0.2, x * y + 0.1))[None]
        area[:, :, 0] = 0
        source_class = (x + 1).reshape(1, 1, *size)
        refl = torch.cat((hdr * 0.1, history_valid, source_class, depth * 2), dim=1)
        frame = Frame(hdr, normal, depth, motion, reactive, history_valid, jitter,
                      (3, 5), reset=step == 0, area_pairs=area, reflection=refl)
        result, previous = model(frame, previous)
        vectors.append({
            "id": f"odd-{step}", "reset": frame.reset, "jitterPx": list(jitter),
            "sourceSize": [3, 2], "displaySize": [5, 3],
            "inputs": {name: tensor_json(value) for name, value in (
                ("hdr", hdr), ("normalRoughness", normal), ("depth", depth), ("motion", motion),
                ("reactive", reactive), ("historyValid", history_valid),
                ("areaPairs", area), ("reflection", refl),
            )},
            "outputs": {name: tensor_json(value) for name, value in result.items()},
        })
    return {"format": "raa.real-frame-golden/1", "layout": "HWC", "f32Absolute": PARITY_ABS,
            "f32Relative": PARITY_REL, "vectors": vectors}


def write_package(output: Path, model, *, kind, seed, trained_heads=(), provenance=None):
    blob = export_weights(model)
    golden = golden_vectors(model)
    golden_data = json_bytes(golden)
    manifest = {
        "format": FORMAT, "modelId": "real-frame-v1", "kind": kind, "seed": seed,
        "weights": "weights.fp32le", "weightsSha256": sha(blob), "weightsByteLength": len(blob),
        "golden": "golden.json", "goldenSha256": sha(golden_data), "tensorLayout": "NHWC",
        "precision": "fp32", "tensorAlignment": 16, "tensors": list(tensor_records()),
        "guideChannels": GUIDES, "trainedHeads": list(trained_heads),
        "cadence": "successfully submitted real frame per view",
        "colorDomain": "nonnegative scene-linear HDR at fixed exposure 1",
        "shippingQualification": "unverified",
    }
    if provenance is not None:
        manifest["provenance"] = provenance
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["packageSha256"] = sha(canonical + blob + golden_data)
    reloaded = load_weights(manifest, blob)
    expected = golden_vectors(reloaded)
    if expected != golden:
        raise ValueError("exported model changed CPU golden tensors")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    (output / "weights.fp32le").write_bytes(blob)
    (output / "golden.json").write_bytes(golden_data)
    return manifest, reloaded
