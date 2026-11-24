"""Validated lossless plane and receipt readers for offline neural training."""

import math
import struct

import torch

from real_frame_model import sha


def read_plane(root, record, channels=None, size=None):
    shape = record["shape"]
    if (record["dtype"] != "f32le" or len(shape) != 3
            or any(type(v) is not int or v < 1 for v in shape)
            or (channels is not None and shape[2] not in channels)
            or (size is not None and tuple(shape[:2]) != size)):
        raise ValueError(f"incompatible plane descriptor {record['file']}")
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"plane escapes its capture directory: {record['file']}")
    data = path.read_bytes()
    length = math.prod(shape)
    if len(data) != length * 4 or record["byteLength"] != len(data) or sha(data) != record["sha256"]:
        raise ValueError(f"damaged plane {record['file']}")
    # frombuffer avoids a Python float for each sample. The manifest fixes the byte order.
    if struct.pack("=I", 1) != b"\x01\x00\x00\x00":
        raise RuntimeError("FP32 plane reader requires a little-endian host")
    tensor = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(*shape)
    if not tensor.isfinite().all():
        raise ValueError(f"nonfinite plane {record['file']}")
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


def require_range(tensor, label, minimum, maximum=None):
    if tensor.min().item() < minimum or (maximum is not None and tensor.max().item() > maximum):
        raise ValueError(f"{label} contains out-of-range values")


def require_windows(manifest):
    source = manifest.get("source", {})
    facts = source.get("facts", manifest.get("facts", {}))
    adapter = facts.get("adapter", {})
    if ("Windows" not in facts.get("browser", "") or facts.get("hardware") is not True
            or adapter.get("isFallbackAdapter") is not False):
        raise ValueError("a hardware Windows capture receipt is required")


def frame_id(record):
    identity = record["id"]
    if not isinstance(identity, str) or not identity:
        raise ValueError("frame IDs must be nonempty strings")
    return identity
