"""Read and write the FP32 capture contract for the optional lighting heads."""

import hashlib
import json
import math
import struct
from pathlib import Path

FORMAT = "raa.optional-head-capture/1"
LAYOUT = dict(rowOrder="top-down", byteOrder="little-endian", exposure=1, jitter=[0, 0])
GUIDES = ("normalX", "normalY", "normalZ", "metricDepth")
POLICY = ("sourceValid", "sourceConfidence", "confidenceValid", "freshlyTraced")
RGB = ("red", "green", "blue")
CHANNELS = {
    "indirect-diffuse": (RGB + GUIDES + POLICY, RGB),
    "indirect-specular": (RGB + GUIDES + ("roughness",) + POLICY, RGB),
    "ao-visibility": (("visibility",) + GUIDES + POLICY, ("visibility",)),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_plane(root, file, shape, values):
    data = struct.pack(f"<{math.prod(shape)}f", *values)
    (root / file).write_bytes(data)
    return dict(file=file, dtype="f32le", shape=shape, byteLength=len(data), sha256=sha(data))


def read_plane(root, desc, shape):
    count = math.prod(shape)
    if desc["dtype"] != "f32le" or desc["shape"] != shape or desc["byteLength"] != count * 4:
        raise ValueError("invalid FP32 plane descriptor")
    file = Path(desc["file"])
    path = (root / file).resolve()
    if file.is_absolute() or ".." in file.parts or not path.is_relative_to(root):
        raise ValueError("plane path must remain inside the capture directory")
    if path.stat().st_size != count * 4:
        raise ValueError(f"capture length mismatch: {file}")
    data = path.read_bytes()
    if len(data) != count * 4 or sha(data) != desc["sha256"]:
        raise ValueError(f"capture length or SHA-256 mismatch: {file}")
    values = struct.unpack(f"<{count}f", data)
    if not all(map(math.isfinite, values)):
        raise ValueError(f"non-finite capture values: {file}")
    return values


def validate_source(source):
    if not isinstance(source["fixture"], str) or not source["fixture"].strip():
        raise ValueError("source fixture must be named")
    if source["kind"] == "cpu-diagnostic":
        return
    if source["kind"] != "windows-render":
        raise ValueError("unknown capture source")
    facts = source["facts"]
    browser = facts["browser"]
    build = source["buildId"]
    if (not isinstance(build, str) or not build.strip() or not isinstance(browser, str)
            or "Windows" not in browser or "Chrome/" not in browser
            or facts["hardware"] is not True or facts["adapter"]["isFallbackAdapter"] is not False):
        raise ValueError("Windows render provenance requires a build ID and Windows Chrome hardware facts")


def signal_seeds(record):
    reference = record["referenceMetadata"]
    batches = reference["seedBatches"]
    if (not isinstance(batches, list) or len(batches) != 2
            or any(not isinstance(batch, list) or not batch for batch in batches)
            or len(batches[0]) != len(batches[1])):
        raise ValueError("reference needs two nonempty equal-size seed batches")
    if not isinstance(reference["estimator"], str) or not reference["estimator"].strip():
        raise ValueError("reference estimator must be named")
    if not isinstance(reference["settings"], dict) or not reference["settings"]:
        raise ValueError("reference estimator settings are required")
    rmse, tolerance = reference["batchRmse"], reference["batchRmseTolerance"]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (rmse, tolerance)):
        raise ValueError("reference batch RMSE and tolerance must be finite numbers")
    if not 0 <= rmse <= tolerance:
        raise ValueError("reference batch RMSE exceeds its tolerance or is negative")
    seeds = record["inputSeeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("at least one input seed is required")
    return seeds + batches[0] + batches[1]


def read_capture(capture, signal):
    data = capture.read_bytes()
    manifest = json.loads(data)
    if manifest["format"] != FORMAT or any(manifest[key] != value for key, value in LAYOUT.items()):
        raise ValueError("expected a top-down, fixed-exposure optional head capture")
    frames = manifest["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("capture needs a nonempty frames array")
    source = manifest["source"]
    validate_source(source)
    root = capture.resolve().parent
    input_channels, output_channels = CHANNELS[signal]
    selected, seen = [], set()
    for frame in frames:
        identity = frame["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("frame IDs must be nonempty and unique")
        seen.add(identity)
        width, height = frame["size"]
        if any(type(d) is not int or d <= 0 for d in (width, height)):
            raise ValueError(f"{identity}: invalid extent")
        if (not isinstance(frame["viewId"], str) or not frame["viewId"]
                or type(frame["frameId"]) is not int or frame["frameId"] < 0
                or type(frame["timeMs"]) not in (int, float) or not math.isfinite(frame["timeMs"])
                or frame["timeMs"] < 0):
            raise ValueError(f"{identity}: invalid view, frame or time")
        signals = frame["signals"]
        if not isinstance(signals, dict) or not signals or signals.keys() - CHANNELS.keys():
            raise ValueError(f"{identity}: expected one or more known signals")
        try:
            seeds = [seed for record in signals.values() for seed in signal_seeds(record)]
            if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
                raise ValueError("expected unsigned 32-bit seeds")
            if len(set(seeds)) != len(seeds):
                raise ValueError("input and reference seed streams must be disjoint across heads within a frame")
            if signal in signals:
                record = signals[signal]
                raw = read_plane(root, record["input"], [height, width, len(input_channels)])
                target = read_plane(root, record["reference"], [height, width, len(output_channels)])
                selected.append((frame, raw, target))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{identity}: {error}") from error
    if not selected:
        raise ValueError(f"capture contains no {signal} frames")
    return source, sha(data), selected
