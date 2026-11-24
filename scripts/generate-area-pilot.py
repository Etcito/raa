#!/usr/bin/env python3
"""Generate the frozen synthetic package and independent scalar golden tensors."""
import hashlib
import json
import math
import struct
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "crates/raa-graph/src/exec_ext/compute/learned"
SHAPES = [(16, 12), (16, 3, 3), (16, 16), (16, 3, 3), (16, 16), (11, 16)]
NAMES = ["stem", "depthwise1", "pointwise1", "depthwise2", "pointwise2", "head"]
SCALES = [1, 1, 1, 1, 1, 1, 1, 0.125, 8, 8, 1, 1]
blob = bytearray()
tensors = []
layers = []


def append(name, shape, values):
    blob.extend(bytes(-len(blob) % 16))
    data = struct.pack(f"<{len(values)}f", *values)
    tensors.append(dict(name=name, dtype="FP32", shape=shape, offset=len(blob),
                        byteLength=len(data), sha256=hashlib.sha256(data).hexdigest()))
    blob.extend(data)
    return struct.unpack(f"<{len(values)}f", data)


for i, (name, shape) in enumerate(zip(NAMES, SHAPES)):
    weights = append(name + ".weight", shape,
                     [((j * 19 + i * 7) % 31 - 15) / 128 for j in range(math.prod(shape))])
    biases = append(name + ".bias", [shape[0]],
                    [(1 + (j + i) % 7) / 64 for j in range(shape[0])])
    layers.append((shape, weights, biases))


def golden(name, width, height, origin, extent, history, hdr, zero):
    ox, oy = origin
    aw, ah = extent
    pixels = [(x, y) for y in range(oy, oy + ah) for x in range(ox, ox + aw)]
    raw = {}
    for x, y in pixels:
        sample = (1 + (x + 3 * y) % 7) * hdr
        current_u = 0 if zero else sample
        raw[x, y] = [current_u * ((x + y) % 5 + 1) / 8, current_u,
                     sample * 0.75, 0 if zero else sample, 0.25, -0.5, 1,
                     sample, 0.03125 * x, -0.0625 * y, float(history), float(x % 2)]
    if zero:
        for p in pixels:
            raw[p][2] = 0

    def tap(x, y, index):
        return (max(ox, min(ox + aw - 1, x + index % 3 - 1)),
                max(oy, min(oy + ah - 1, y + index // 3 - 1)))

    activations = {p: [v * s if history or c not in (2, 3) else 0
                       for c, (v, s) in enumerate(zip(raw[p], SCALES))] for p in pixels}
    for stage, (shape, weights, biases) in enumerate(layers):
        out = {}
        for p in pixels:
            values = []
            for c, bias in enumerate(biases):
                if len(shape) == 3:
                    value = bias + sum(weights[c * 9 + k] * activations[tap(*p, k)][c]
                                       for k in range(9))
                else:
                    value = bias + sum(weights[c * shape[1] + k] * v
                                       for k, v in enumerate(activations[p]))
                values.append(max(0, value) if stage < 5 else value)
            out[p] = values
        activations = out

    def sigmoid(x):
        return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

    expected = []
    inputs = []
    for y in range(height):
        for x in range(width):
            p = (x, y)
            inputs.extend(raw.get(p, [-123] * 12))
            if p not in raw:
                expected.extend([0] * 12)
                continue
            logits = activations[p]
            exp = [math.exp(v - max(logits[:9])) for v in logits[:9]]
            spatial = [v / sum(exp) for v in exp]
            h = sigmoid(logits[9]) if history else 0
            pair = [(1 - h) * sum(w * raw[tap(x, y, k)][c] for k, w in enumerate(spatial))
                    + h * raw[p][c + 2] for c in range(2)]
            expected.extend([*spatial, h, sigmoid(logits[10]),
                             pair[0] / pair[1] if pair[1] > 0 else 1])
    return dict(name=name, size=[width, height], origin=origin, extent=extent,
                input=inputs, output=expected)


vectors = [
    golden("single", 1, 1, [0, 0], [1, 1], True, 1, False),
    golden("odd-offset", 9, 5, [1, 1], [8, 4], True, 1, False),
    golden("rejected", 3, 1, [0, 0], [3, 1], False, 1, False),
    golden("high-hdr", 3, 3, [0, 0], [3, 3], False, 1024, False),
    golden("zero-denominator", 1, 1, [0, 0], [1, 1], True, 1, True),
]
golden_bytes = (json.dumps(vectors, separators=(",", ":"), allow_nan=False) + "\n").encode()
manifest = dict(
    operatorAbi="raa.area-inference/1", signalSchema="raa.area-pair-guides/1", dtype="FP32",
    layout="NHWC", byteOrder="little-endian", alignment=16, workgroup=[8, 8, 1],
    channels=["currentS", "currentU", "acceptedPreviousS", "acceptedPreviousU",
              "viewNormalX", "viewNormalY", "viewNormalZ", "metricDepth",
              "unjitteredMotionUVX", "unjitteredMotionUVY", "historyValid", "lightChange"],
    normalizationScales=SCALES, supportedScales=[1], rayBudgets=[1], exposureRange=[1, 1],
    requiredGuides=["viewNormal", "metricDepth", "unjitteredMotionUV", "historyValid", "lightChange"],
    cadence="real-frame", outputs=["spatialSoftmax[9]", "maskedHistoryGate", "confidence", "visibility"],
    tolerance=dict(absolute=1e-5, relative=1e-5), tensors=tensors,
    weightsSha256=hashlib.sha256(blob).hexdigest(),
    golden=dict(file="golden.json", sha256=hashlib.sha256(golden_bytes).hexdigest(),
                cases=[v["name"] for v in vectors]),
)
canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
manifest["packageSha256"] = hashlib.sha256(canonical + blob + golden_bytes).hexdigest()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="compare every byte to the frozen package")
args = parser.parse_args()
for name, data in [
    ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()),
    ("weights.fp32le", blob), ("golden.json", golden_bytes),
]:
    path = ROOT / name
    if args.check:
        if path.read_bytes() != data:
            raise SystemExit(f"frozen package differs: {name}")
    else:
        path.write_bytes(data)
