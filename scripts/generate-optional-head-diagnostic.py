#!/usr/bin/env python3
"""Write a small deterministic CPU diagnostic capture for optional head training."""

import argparse
import math
import random
from pathlib import Path

from optional_head_capture import CHANNELS, FORMAT, LAYOUT, json_bytes, write_plane


def pixels(width, height, frame):
    for y in range(height):
        for x in range(width):
            u, v = (x + 0.5) / width, (y + 0.5) / height
            depth = 0 if (x, y) == (0, 0) else 1 + 2 * u + v + 0.25 * frame
            normal = (u - 0.5, v - 0.5, math.sqrt(1 - (u - 0.5)**2 - (v - 0.5)**2))
            yield depth, normal, 0.15 + 0.7 * v


NOISE = {"indirect-diffuse": 0.5, "indirect-specular": 0.45, "ao-visibility": 0.28}


def sample(signal, geometry, seed):
    rng = random.Random(seed)
    ao = signal == "ao-visibility"
    amplitude = NOISE[signal]
    for depth, (nx, ny, _), roughness in geometry:
        if signal == "indirect-diffuse":
            clean = (1.25 + nx, 0.55 + 0.2 * depth, 0.65 + 0.4 * ny)
        elif signal == "indirect-specular":
            clean = (0.4 + 1.2 * (1 - roughness), 0.7 + 0.2 * depth, 0.55 + 0.25 * nx)
        else:
            clean = (0.18 + 0.15 * depth + 0.16 * nx,)
        for value in clean:
            noisy = max(0, value + rng.uniform(-amplitude, amplitude))
            if depth == 0:
                yield 1.0 if ao else 0.0
            else:
                yield min(noisy, 1.0) if ao else noisy


def average(samples):
    return [sum(lanes) / len(lanes) for lanes in zip(*samples)]


def generate(output):
    output.mkdir()
    frames = []
    for frame_id, (width, height) in enumerate(((7, 5), (5, 3))):
        identity = f"diagnostic-{frame_id}"
        geometry = list(pixels(width, height, frame_id))
        signals = {}
        for index, (signal, (inputs, outputs)) in enumerate(CHANNELS.items()):
            base = (frame_id + 1) * 1000 + index * 100
            batches = [list(range(base + start, base + start + 8)) for start in (10, 30)]
            means = [average([list(sample(signal, geometry, seed)) for seed in batch]) for batch in batches]
            target = average(means)
            count = len(outputs)
            rmse = math.sqrt(sum((a - b)**2 for a, b in zip(*means)) / ((width * height - 1) * count))
            noisy = list(sample(signal, geometry, base))
            raw = []
            for pixel, (depth, normal, roughness) in enumerate(geometry):
                source_valid = depth > 0 and pixel % 5 != 0
                confidence_valid = source_valid and pixel % 4 != 0
                freshly_traced = source_valid and pixel % 3 == 0
                current = noisy[pixel * count:(pixel + 1) * count]
                if not source_valid:
                    current = [1.0] if count == 1 else [0.0] * count
                raw.extend(current)
                raw.extend(normal if depth > 0 else (0, 0, 0))
                raw.append(depth)
                if signal == "indirect-specular":
                    raw.append(roughness if depth > 0 else 0)
                confidence = (0.9 if freshly_traced else 0.4) if confidence_valid else 0.0
                raw.extend((source_valid, confidence, confidence_valid, freshly_traced))
            prefix = f"{identity}-{signal}"
            signals[signal] = dict(
                inputSeeds=[base],
                input=write_plane(output, f"{prefix}-input.fp32le", [height, width, len(inputs)], raw),
                reference=write_plane(output, f"{prefix}-reference.fp32le", [height, width, count], target),
                referenceMetadata=dict(estimator="synthetic-noisy-analytic-mean/1", seedBatches=batches,
                                       settings=dict(distribution="clipped-uniform", noiseAmplitude=NOISE[signal]),
                                       batchRmse=rmse, batchRmseTolerance=0.25),
            )
        frames.append(dict(id=identity, viewId="diagnostic", frameId=frame_id, timeMs=16 * frame_id,
                           size=[width, height], signals=signals))
    manifest = dict(format=FORMAT, source=dict(kind="cpu-diagnostic", fixture="analytic-grid/1"),
                    **LAYOUT, frames=frames)
    (output / "manifest.json").write_bytes(json_bytes(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the diagnostic manifest and planes")
    args = parser.parse_args()
    try:
        generate(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
