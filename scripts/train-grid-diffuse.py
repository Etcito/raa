#!/usr/bin/env python3
"""Fit and score the live spherical-grid diffuse interpolation gate."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES = (
    "deltaCellX", "deltaCellY", "deltaCellZ", "normalCosine", "luminanceMomentCosine",
    "flooredTrilinear", "surfaceSupport", "valid",
)
LAYOUT = (("w1", (8, 8)), ("b1", (8,)), ("w2", (8,)), ("b2", (1,)))
FORMAT = "raa.spherical-diffuse-gate/1"
UNITS = "scene-linear irradiance/pi before receiver albedo"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class Taps:
    features: np.ndarray
    weights: np.ndarray
    directional: np.ndarray
    legacy: np.ndarray
    teacher_a: np.ndarray
    teacher_b: np.ndarray
    blend: np.ndarray
    legacy_blend: np.ndarray

    @property
    def supported(self):
        return np.count_nonzero(self.weights > 0, axis=1) >= 2

    @property
    def teacher(self):
        return (self.teacher_a + self.teacher_b) * np.float32(0.5)

    def take(self, selection):
        return Taps(*(v[selection] for v in vars(self).values()))


def pack(params):
    return b"".join(np.asarray(value, dtype="<f4").tobytes(order="C") for value in params)


def unpack(blob):
    if len(blob) != 81 * 4:
        raise ValueError("Expected 81 float32 weights")
    words = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(words).all():
        raise ValueError("Model contains non-finite weights")
    arrays = []
    offset = 0
    for _, shape in LAYOUT:
        size = int(np.prod(shape))
        arrays.append(words[offset:offset + size].reshape(shape))
        offset += size
    return tuple(arrays)


def gated_mix(taps, params=None, require_support=True):
    weight = taps.weights
    if params is not None:
        w1, b1, w2, b2 = params
        hidden = np.maximum(taps.features @ w1.T + b1, 0)
        score = hidden @ w2 + b2
        weight = weight * np.exp(np.clip(score, -4, 4))
    denom = weight.sum(axis=1, keepdims=True)
    mixture = weight / np.where(denom > 0, denom, 1)
    rgb = (mixture[..., None] * taps.directional).sum(axis=1)
    if require_support:
        rgb[~taps.supported] = 0
    return rgb


def loss_gradient(taps, params, scale):
    w1, b1, w2, b2 = params
    activations = taps.features @ w1.T + b1
    hidden = np.maximum(activations, 0)
    logit = hidden @ w2 + b2
    weight = taps.weights * np.exp(np.clip(logit, -4, 4))
    mixture = weight / weight.sum(axis=1, keepdims=True)
    prediction = (mixture[..., None] * taps.directional).sum(axis=1)
    error = np.log1p(prediction / scale) - np.log1p(taps.teacher / scale)
    dp = (2.0 / error.size) * error / (scale + prediction)
    # Differentiating the normalization keeps zero-mass and empty padding out of the fit.
    ds = mixture * ((taps.directional - prediction[:, None, :]) * dp[:, None, :]).sum(axis=-1)
    ds *= (logit > -4) & (logit < 4)
    dh = ds[..., None] * w2 * (activations > 0)
    return float(np.mean(error ** 2)), (
        dh.reshape(-1, 8).T @ taps.features.reshape(-1, 8),
        dh.sum(axis=(0, 1)),
        (ds[..., None] * hidden).sum(axis=(0, 1)),
        np.array([ds.sum()], dtype=np.float32),
    )


def log_error(prediction, target, scale):
    return float(np.mean((np.log1p(prediction / scale) - np.log1p(target / scale)) ** 2))


def optimize(taps, steps, batch_size, seed, learning_rate):
    taps = taps.take(taps.supported)
    if not len(taps.weights):
        raise ValueError("Training contains no hit with at least two supported cells")
    generator = np.random.default_rng(seed)
    params = [
        generator.normal(0, np.sqrt(2 / 8), (8, 8)).astype(np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    ]
    target = taps.teacher
    positive = target @ LUMA
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("Training teacher has no nonzero diffuse target")
    scale = float(np.median(positive))
    best = [p.copy() for p in params]
    best_loss = log_error(gated_mix(taps, params), target, scale)
    history = [{"step": 0, "logMse": best_loss}]
    m = [np.zeros_like(value) for value in params]
    v = [np.zeros_like(value) for value in params]
    batch_size = min(batch_size, len(target))
    for step in range(1, steps + 1):
        mini = taps.take(generator.choice(len(target), batch_size, replace=False))
        loss, gradients = loss_gradient(mini, params, scale)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
            raise ValueError(f"Non-finite optimization at step {step}")
        rate = learning_rate * np.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
        for weight, first, second, grad in zip(params, m, v, gradients):
            first *= 0.9
            first += 0.1 * grad
            second *= 0.999
            second += 0.001 * grad ** 2
            weight -= rate * first / (np.sqrt(second) + 1e-8)
        if step % 50 == 0 or step == steps:
            train_loss = log_error(gated_mix(taps, params), target, scale)
            history.append({"step": step, "logMse": train_loss})
            if train_loss < best_loss:
                best_loss = train_loss
                best = [p.copy() for p in params]
    return tuple(best), scale, history


def metrics(prediction, target, scale):
    delta = (prediction - target).astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    target_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
    return {
        "rgbRmse": rmse,
        "rgbNrmse": rmse / target_rms if target_rms else None,
        "rgbMae": float(np.mean(np.abs(delta))),
        "logRgbRmse": float(np.sqrt(log_error(prediction, target, scale))),
        "meanLumaBias": float(np.mean(delta @ LUMA)),
        "meanLuma": float(np.mean(prediction @ LUMA)),
    }


def score(taps, params, scale):
    mask = taps.supported
    result = {"hits": len(mask), "supportedHits": int(mask.sum()), "coverage": float(mask.mean())}
    analytic = gated_mix(taps)
    learned = gated_mix(taps, params)
    # Scores of the cache contribution on all requests retain weak or absent support.
    target = taps.teacher
    result["allRequestCacheContribution"] = {
        "legacy": metrics(taps.legacy * taps.legacy_blend[:, None], target, scale),
        "directionalAnalytic": metrics(analytic * taps.blend[:, None], target, scale),
        "directionalLearned": metrics(learned * taps.blend[:, None], target, scale),
    }
    if not mask.any():
        return result
    taps = taps.take(mask)
    target = taps.teacher
    prediction = learned[mask]
    analytic = analytic[mask]
    exists = taps.weights[..., None] > 0
    low = np.where(exists, taps.directional, np.inf).min(axis=1)
    high = np.where(exists, taps.directional, -np.inf).max(axis=1)
    bound = np.clip(target, low, high)
    result.update({
        "legacy": metrics(taps.legacy, target, scale),
        "directionalAnalytic": metrics(analytic, target, scale),
        "directionalLearned": metrics(prediction, target, scale),
        "referenceBatch": metrics(taps.teacher_a, taps.teacher_b, scale),
        "channelIntervalBound": metrics(bound, target, scale),
        "targetOutsideChannelInterval": float(np.any(bound != target, axis=1).mean()),
    })
    return result


CAPTURE_FORMAT = "raa.spherical-diffuse-capture/1"


def is_sha256(text):
    return isinstance(text, str) and len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def relative_read(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Capture file escapes its directory: {name}")
    return path.read_bytes()


def unpack_rows(data, row_count):
    if type(row_count) is not int or row_count <= 0:
        raise ValueError("Invalid capture row count")
    if len(data) != row_count * 128 * 4:
        raise ValueError("Expected 128 words per native capture row")
    rows = np.frombuffer(data, dtype="<f4").reshape(row_count, 128)
    words = rows.view("<u4")
    cells = rows[:, 32:].reshape(row_count, 8, 12)
    features, directional, weights = cells[:, :, :8], cells[:, :, 8:11], cells[:, :, 11]
    legacy, a, b = rows[:, 20:23], rows[:, 12:15], rows[:, 16:19]
    blend, legacy_blend = rows[:, 15], rows[:, 19]
    rgb = (directional, legacy, a, b, rows[:, 24:27], rows[:, 28:31])
    geometry = (rows[:, :3], rows[:, 4:7], rows[:, 8:11])
    if not all(np.isfinite(value).all() for value in (features, weights, blend, legacy_blend, *geometry, *rgb)):
        raise ValueError("Capture contains non-finite input or target")
    if any((value < 0).any() for value in rgb):
        raise ValueError("Pre-albedo irradiance cannot be negative")
    if any(((value < 0) | (value > 1)).any() for value in (weights, blend, legacy_blend)):
        raise ValueError("Weight or support blend is outside [0, 1]")
    if not np.array_equal(features[..., 7], weights > 0):
        raise ValueError("Validity must agree with positive base weight")
    if not np.array_equal(np.count_nonzero(weights > 0, axis=1), words[:, 11]):
        raise ValueError("Positive weights disagree with native neighbor count")
    if not np.allclose(weights, features[..., 5] * features[..., 6], atol=1e-8, rtol=2e-6):
        raise ValueError("Base weight disagrees with trilinear × surface support")
    valid = features[weights > 0]
    if ((np.abs(valid[:, :3]) > 2.5001).any() or
        (np.abs(valid[:, 3:5]) > 1.0001).any() or
        ((valid[:, 5:7] <= 0) | (valid[:, 5:7] > 1.0001)).any()):
        raise ValueError("Neighbor features disagree with the runtime ABI")
    taps = Taps(features, weights, directional, legacy, a, b, blend, legacy_blend)
    if (blend[~taps.supported] > 0).any() or (legacy_blend[~taps.supported] > 0).any():
        raise ValueError("An unsupported hit has nonzero cache blend")
    if not np.allclose(rows[:, 28:31], gated_mix(taps), atol=4e-5, rtol=2e-5):
        raise ValueError("Analytic mixture does not match the GPU readback")
    return taps, rows


def read_capture(path, split):
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest["format"] != CAPTURE_FORMAT or manifest["units"] != UNITS:
        raise ValueError(f"Unknown capture schema or units: {path}")
    scene = manifest["scene"]
    if scene["split"] != split or not scene["id"] or not is_sha256(scene["geometrySha256"]):
        raise ValueError(f"Capture is not an identified {split} scene: {path}")
    if manifest["source"]["platform"] != "win32" or not manifest["source"]["adapter"]:
        raise ValueError(f"Capture lacks native Windows GPU provenance: {path}")
    if not manifest["frames"]:
        raise ValueError(f"No frames in capture: {path}")
    identifiers = set()
    frames = []
    for record in manifest["frames"]:
        if not record["id"] or record["id"] in identifiers:
            raise ValueError(f"Missing or repeated frame identity: {record['id']}")
        identifiers.add(record["id"])
        ref = record["reference"]
        if ref["independent"] is not True or not is_sha256(ref["firstHitSha256"]) or ref["samplesPerBank"] <= 0:
            raise ValueError(f"Unpaired reference banks: {record['id']}")
        data = relative_read(path.parent, record["file"])
        if sha256(data) != record["sha256"]:
            raise ValueError(f"Capture digest mismatch: {record['file']}")
        taps, rows = unpack_rows(data, record["rows"])
        pixel_indices = None
        if "rasterIndices" in record:
            raster = record["rasterIndices"]
            indices = relative_read(path.parent, raster["file"])
            if len(indices) != 4 * record["rows"] or sha256(indices) != raster["sha256"]:
                raise ValueError(f"Raster index digest or length mismatch: {raster['file']}")
            pixel_indices = np.frombuffer(indices, dtype="<u4")
        frames.append((record, taps, rows, pixel_indices))
    return manifest, frames, sha256(payload)


def concatenate(frames):
    if len(frames) == 1:
        return frames[0][1]
    return Taps(*(np.concatenate(parts) for parts in zip(*(vars(frame[1]).values() for frame in frames))))


def coverage(taps, params):
    blend = taps.blend
    counts = np.count_nonzero(taps.weights > 0, axis=1)
    distribution = np.bincount(counts, minlength=9)
    singles = taps.take(counts == 1)
    return {
        "neighborCountHistogram0To8": [int(count) for count in distribution],
        "zeroNeighbor": int(distribution[0]),
        "oneNeighbor": int(distribution[1]),
        "fallbackFull": int((blend == 0).sum()),
        "fallbackPartial": int(((blend > 0) & (blend < 1)).sum()),
        "fallbackFraction": float((blend < 1).mean()),
        "meanFallbackBlend": float((1 - blend).mean()),
        # Check the weighting itself, even though the runtime routes one-cell hits to fallback.
        "oneNeighborNormalizedRgbDeltaMax": float(np.max(np.abs(
            gated_mix(singles, require_support=False)
            - gated_mix(singles, params, require_support=False)
        ))) if len(singles.weights) else None,
    }


def score_frames(frames, params, scale):
    taps = concatenate(frames)
    result = score(taps, params, scale)
    result.update(coverage(taps, params))
    return result


def residuals(taps, params):
    target = taps.teacher
    return np.stack((
        taps.legacy * taps.legacy_blend[:, None] - target,
        gated_mix(taps) * taps.blend[:, None] - target,
        gated_mix(taps, params) * taps.blend[:, None] - target,
    ), axis=1)


def unique_indices(keys):
    unique, index, counts = np.unique(keys, return_index=True, return_counts=True)
    return unique[counts == 1], index[counts == 1]


def temporal_pairs(frames, params):
    previous = {}
    result = []
    for record, taps, rows, raster in frames:
        trajectory = record.get("trajectory")
        if trajectory is None:
            continue
        words = rows.view("<u4")
        key_kind = "raster" if raster is not None else "worldHitHash"
        keys, indices = unique_indices(raster if raster is not None else words[:, 31])
        primitives = words[:, 3][indices]
        valid = taps.supported[indices]
        current = residuals(taps, params)[indices]
        old = previous.get(trajectory)
        previous[trajectory] = (record, key_kind, keys, primitives, valid, current)
        if old is None:
            continue
        old_record, old_kind, old_keys, old_primitives, old_supported, old_residuals = old
        if old_kind != key_kind:
            raise ValueError(f"Trajectory matching key changed: {trajectory}")
        common, left, right = np.intersect1d(old_keys, keys, assume_unique=True, return_indices=True)
        supported = old_supported[left] & valid[right]
        delta = current[right] - old_residuals[left]
        same = old_primitives[left] == primitives[right]
        groups = {"all": np.ones(common.shape, dtype=bool), "supportedBoth": supported,
                  "sameTriangle": same, "differentTriangle": ~same}
        scores = {}
        for name, mask in groups.items():
            scores[name] = {"pairs": int(mask.sum())}
            if mask.any():
                errors = np.sqrt(np.mean(delta[mask].astype(np.float64) ** 2, axis=(0, 2)))
                scores[name]["residualChangeRgbRmse"] = dict(zip(
                    ("legacy", "directionalAnalytic", "directionalLearned"), errors.tolist()
                ))
        result.append({"trajectory": trajectory, "from": old_record["id"], "to": record["id"],
                       "matchKey": key_kind, "commonRequests": int(common.size), **scores})
    return result


def native_parity(record, taps, rows, params, weights_sha):
    if weights_sha is None or not record.get("learnedWeightsSha256"):
        return None
    if record["learnedWeightsSha256"] != weights_sha:
        raise ValueError(f"Native replay used different weights: {record['id']}")
    bits = taps.supported
    actual = rows[:, 24:27][bits]
    expected = gated_mix(taps, params)[bits]
    if not np.allclose(actual, expected, atol=4e-5, rtol=2e-5):
        raise ValueError(f"Native learned irradiance disagrees with the frozen checkpoint: {record['id']}")
    return {"rows": int(bits.sum()), "maxRgbAbsoluteDelta": float(np.max(np.abs(actual - expected))) if bits.any() else None}


def score_scene(manifest, frames, params, scale, weights_sha=None):
    groups = {}
    for frame in frames:
        groups.setdefault(frame[0]["condition"], []).append(frame)
    return {
        "scene": manifest["scene"],
        "source": manifest["source"],
        "all": score_frames(frames, params, scale),
        "conditions": {name: score_frames(group, params, scale) for name, group in groups.items()},
        "frames": [
            {**{k: record[k] for k in ("id", "condition", "reference", "sha256")},
             **{k: record[k] for k in ("trajectory", "camera", "lighting", "gpuTimings") if k in record},
             **score_frames([frame], params, scale),
             "nativeLearnedParity": native_parity(record, frame[1], frame[2], params, weights_sha)}
            for frame in frames for record in (frame[0],)
        ],
        "temporalPairs": temporal_pairs(frames, params),
    }


def fit(args):
    from datetime import UTC, datetime

    capture, frames, digest = read_capture(args.capture, "train")
    taps = concatenate(frames)
    params, scale, history = optimize(taps, args.steps, args.batch, args.seed, args.learning_rate)
    weights = pack(params)
    if np.count_nonzero(params[2]) == 0:
        raise ValueError("Optimizer produced no learned reweighting")
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    weight_path = directory / "weights.fp32le"
    weight_path.write_bytes(weights)
    model = {
        "format": FORMAT,
        "frozenAtUtc": datetime.now(UTC).isoformat(),
        "weights": {"file": weight_path.name, "sha256": sha256(weights), "bytes": len(weights)},
        "features": FEATURES,
        "tensors": [{"name": name, "shape": shape} for name, shape in LAYOUT],
        "training": {
            "scene": capture["scene"],
            "captureSha256": digest,
            "steps": args.steps,
            "batchSize": args.batch,
            "seed": args.seed,
            "learningRate": args.learning_rate,
            "logScale": scale,
        },
        "protocol": "sdfm:f8a8893ab4a1f30093317c9dbc09d27c2bfdb9da",
    }
    write_json(directory / "manifest.json", model)
    frozen_sha = sha256((directory / "manifest.json").read_bytes())
    write_json(directory / "freeze.json", {
        "weightsSha256": model["weights"]["sha256"], "manifestSha256": frozen_sha,
        "frozenAtUtc": model["frozenAtUtc"], "trainingCaptureSha256": digest,
    })
    print(f"Frozen weights {model['weights']['sha256']}", flush=True)
    write_json(directory / "train.json", {
        "weightsSha256": model["weights"]["sha256"], "history": history,
        **score_scene(capture, frames, unpack(weights), scale),
    })


def evaluate(args):
    model_payload = (args.model / "manifest.json").read_bytes()
    model = json.loads(model_payload)
    freeze = json.loads((args.model / "freeze.json").read_bytes())
    if model["format"] != FORMAT or sha256(model_payload) != freeze["manifestSha256"]:
        raise ValueError("Checkpoint does not match its freeze receipt")
    weight_data = relative_read(args.model, model["weights"]["file"])
    digest = sha256(weight_data)
    if digest != args.weights_sha256 or digest != freeze["weightsSha256"] or digest != model["weights"]["sha256"]:
        raise ValueError("Checkpoint differs from the pre-registered weight hash")
    params = unpack(weight_data)
    training = model["training"]
    seen_id = {training["scene"]["id"]}
    seen_geom = {training["scene"]["geometrySha256"]}
    reserved = []
    for path in args.capture:
        manifest, frames, capture_sha = read_capture(path, "test")
        scene = manifest["scene"]
        if scene["id"] in seen_id or scene["geometrySha256"] in seen_geom or training["captureSha256"] == capture_sha:
            raise ValueError(f"Overlapping training or reserved scene: {scene['id']}")
        seen_id.add(scene["id"])
        seen_geom.add(scene["geometrySha256"])
        reserved.append({"captureSha256": capture_sha, **score_scene(manifest, frames, params, training["logScale"], digest)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump({"weightsSha256": digest, "freeze": freeze, "reserved": reserved}, out, indent=2, allow_nan=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="Fit only one training room and freeze the result")
    fit_parser.add_argument("capture", type=Path)
    fit_parser.add_argument("output", type=Path)
    fit_parser.add_argument("--steps", type=int, default=1500)
    fit_parser.add_argument("--batch", type=int, default=512)
    fit_parser.add_argument("--seed", type=int, default=20261125)
    fit_parser.add_argument("--learning-rate", type=float, default=0.01)
    fit_parser.set_defaults(run=fit)
    eval_parser = commands.add_parser("score", help="Read only held-out scenes with the required frozen hash")
    eval_parser.add_argument("model", type=Path)
    eval_parser.add_argument("output", type=Path)
    eval_parser.add_argument("capture", nargs="+", type=Path)
    eval_parser.add_argument("--weights-sha256", required=True)
    eval_parser.set_defaults(run=evaluate)
    args = parser.parse_args()
    if args.command == "fit" and (args.steps < 1 or args.batch < 1 or not 0 < args.learning_rate < 1):
        parser.error("Expected positive steps and batch size and a learning rate in (0, 1)")
    args.run(args)


if __name__ == "__main__":
    main()
