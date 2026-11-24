#!/usr/bin/env python3
"""Fit and export the causal real-frame FP32 ABI on lossless Windows captures."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from real_frame_model import (
    Frame, at, configure_cpu, display_coordinates, export_weights, grid, json_bytes, load_weights,
    sample_bilinear, seeded, sha, visibility_ratio, write_package,
)


from neural_capture import frame_id, read_plane, require_range, require_windows


CAPTURE_FORMAT = "raa.gallery-temporal-sequence/1"


@dataclass
class Sample:
    id: str
    frame: Frame
    target: torch.Tensor
    area_target: torch.Tensor | None
    reflection_target: torch.Tensor | None
    analytic: torch.Tensor | None
    fingerprint: tuple
    raster_target: torch.Tensor | None = None
    event: bool = False
    disocclusion: torch.Tensor | None = None


@dataclass
class Sequence:
    id: str
    group: str
    split: str
    samples: list
    replay: str | None = None


def load_capture(path, accepted=None):
    data = path.read_bytes()
    manifest = json.loads(data)
    if manifest["format"] != CAPTURE_FORMAT:
        raise ValueError(f"expected {CAPTURE_FORMAT}")
    if manifest.get("rowOrder", "top-down") != "top-down" or manifest.get("byteOrder", "little-endian") != "little-endian":
        raise ValueError("capture must be top-down little-endian")
    if manifest.get("exposure", 1) != 1:
        raise ValueError("capture HDR must use exposure 1")
    require_windows(manifest)
    sequences = []
    sequence_ids = set()
    occupied_groups = {}
    all_ids = set()
    fingerprints = {}
    for sequence in manifest["sequences"]:
        seq_id = sequence["id"]
        group = sequence.get("groupId", seq_id)
        split = {"train": "train", "dev": "development", "development": "development",
                 "test": "held-out", "held-out": "held-out"}.get(sequence["split"])
        if not isinstance(seq_id, str) or not seq_id or seq_id in sequence_ids or not isinstance(group, str) or not group or split is None:
            raise ValueError("sequence IDs and split assignments must be explicit and unique")
        if occupied_groups.setdefault(group, split) != split:
            raise ValueError(f"split leakage for capture group {group}")
        sequence_ids.add(seq_id)
        if accepted is not None and split not in accepted:
            continue
        samples = []
        previous_id = None
        previous_view = None
        previous_time = None
        for index, record in enumerate(sequence["frames"]):
            identity = frame_id(record)
            if identity in all_ids:
                raise ValueError(f"duplicate capture frame ID {identity}")
            all_ids.add(identity)
            step = record["frameId"]
            stamp = record["timeMs"]
            view = record.get("viewId", "main")
            if type(step) is not int or step < 0 or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError(f"invalid frame clock {identity}")
            reset = record.get("reset", index == 0)
            if type(reset) is not bool:
                raise ValueError(f"invalid reset flag {identity}")
            if index and not reset and (step != previous_id + 1 or view != previous_view or stamp <= previous_time):
                raise ValueError(f"nonconsecutive frames without an explicit reset at {identity}")
            previous_id, previous_view, previous_time = step, view, stamp
            p = record["planes"]
            hdr = read_plane(path.parent, p["hdr"], (3,))
            size = hdr.shape[-2:]

            def scene_plane(name, channels):
                return read_plane(path.parent, p[name], channels, size)

            guide = scene_plane("guide", (12,))
            depth = scene_plane("depth", (1,))
            motion = scene_plane("motion", (3, 4))[:, :3]
            reactive = guide[:, 8:9]
            history_valid = guide[:, 7:8]
            jitter = tuple(record.get("jitterPx", guide[0, 9:11, 0, 0].tolist()))
            if len(jitter) != 2 or not all(math.isfinite(value) for value in jitter):
                raise ValueError(f"invalid jitter at {identity}")
            for coordinate, value in enumerate(jitter):
                if not torch.allclose(guide[:, 9 + coordinate], torch.full_like(depth[:, 0], value),
                                      rtol=0.0, atol=1e-6):
                    raise ValueError(f"jitter disagrees with guide at {identity}")
            target = read_plane(path.parent, p["label"], (3,))
            raster = read_plane(path.parent, p["rasterLabel"], (3,), target.shape[-2:]) if "rasterLabel" in p else None
            composite = scene_plane("compositeHdr", (3,)) if "compositeHdr" in p else None
            if raster is not None and composite is None:
                raise ValueError(f"matched raster supervision needs the current SSR composite at {identity}")
            masks = scene_plane("masks", (4,)) if "masks" in p else None
            analytic = read_plane(path.parent, p["analyticDisplayHdr"], (3,), target.shape[-2:]) if "analyticDisplayHdr" in p else None
            for image, label in ((hdr, "HDR"), (target, "teacher HDR"), (depth, "metric depth")):
                require_range(image, label, 0.0)
            for image, label in ((reactive, "reactive"), (history_valid, "history validity"), (guide[:, 4:5], "roughness")):
                require_range(image, label, 0.0, 1.0)
            if analytic is not None:
                require_range(analytic, "analytic HDR", 0.0)
            if raster is not None:
                require_range(raster, "matched raster HDR", 0.0)
            if composite is not None:
                require_range(composite, "scene SSR composite", 0.0)
            fingerprint = tuple(p[name]["sha256"] for name in ("hdr", "guide", "label"))
            if fingerprints.setdefault(fingerprint, split) != split:
                raise ValueError(f"identical source, guides and target appear across splits at {identity}")

            areas = scene_plane("areaPairs", (2,)) if "areaPairs" in p else None
            area_target = scene_plane("areaVisibility", (1,)) if "areaVisibility" in p else None
            reflections = scene_plane("reflection", (6,)) if "reflection" in p else None
            reflection_target = scene_plane("reflectionTarget", (3,)) if "reflectionTarget" in p else None
            if area_target is not None:
                if areas is None:
                    raise ValueError(f"missing area input for label {identity}")
                require_range(area_target, "area visibility", 0.0, 1.0)
            if areas is not None:
                require_range(areas, "area samples", 0.0)
            if reflection_target is not None:
                if reflections is None:
                    raise ValueError(f"missing reflection input for label {identity}")
                require_range(reflection_target, "reflection target", 0.0)
            if reflections is not None:
                require_range(reflections, "reflection input", 0.0)
                require_range(reflections[:, 3:4], "reflection validity", 0.0, 1.0)
                require_range(reflections[:, 4:5], "reflection source class", 0.0, 4.0)
            normal_roughness = torch.cat((guide[:, :3], guide[:, 4:5]), dim=1)
            frame = Frame(hdr, normal_roughness, depth, motion, reactive, history_valid,
                          jitter, target.shape[-2:], reset, areas, reflections, composite)
            samples.append(Sample(identity, frame, target, area_target, reflection_target,
                                  analytic, fingerprint, raster, bool(record.get("event")),
                                  masks[:, :1] > 0.5 if masks is not None else None))
        if len(samples) < 2:
            raise ValueError(f"sequence {seq_id} needs at least two real frames")
        sequences.append(Sequence(seq_id, group, split, samples, sequence.get("replaySha256")))
    if not sequences:
        raise ValueError("capture has no sequences")
    return sequences, {"sha256": sha(data), "source": manifest["source"]}


def load_area_capture(path):
    data = path.read_bytes()
    manifest = json.loads(data)
    if manifest["format"] != "raa.area-pilot-capture/1" or not manifest["frames"]:
        raise ValueError("expected a nonempty Ticket 10 area capture")
    require_windows(manifest)
    if manifest["jitter"] != [0, 0] or manifest["exposure"] != 1:
        raise ValueError("Ticket 10 area uses fixed unit exposure and zero jitter")
    result = []
    for record in manifest["frames"]:
        inputs = read_plane(path.parent, record["input"], (12,))
        reference = record["reference"]
        if len(reference["shape"]) == 2:
            reference = {**reference, "shape": [*reference["shape"], 1]}
        ref = read_plane(path.parent, reference, (1,), inputs.shape[-2:])
        pairs = inputs[:, :2]
        normal = torch.cat((inputs[:, 4:7], torch.zeros_like(ref)), dim=1)
        motion = torch.cat((inputs[:, 8:10], 1 - inputs[:, 10:11]), dim=1)
        frame = Frame(torch.zeros_like(normal[:, :3]), normal, inputs[:, 7:8], motion,
                      inputs[:, 11:12], inputs[:, 10:11], (0, 0), ref.shape[-2:],
                      True, pairs)
        result.append((frame_id(record), frame, ref))
    return result, {"sha256": sha(data), "source": manifest.get("source"), "fixture": manifest["fixture"]}


def masked_mean(value, mask):
    return (value * mask).sum() / (mask.sum().clamp_min(1) * value.shape[1])


def area_loss(prediction, reference, pairs):
    return masked_mean((prediction - reference).square(), pairs[:, 1:2] > 1e-8)


def charbonnier(delta):
    return (delta.square() + 1e-4).sqrt() - 0.01


def edge_error(delta):
    errors = []
    if delta.shape[-1] > 1:
        errors.append(delta.diff(dim=-1).abs().mean())
    if delta.shape[-2] > 1:
        errors.append(delta.diff(dim=-2).abs().mean())
    return sum(errors) / max(1, len(errors))


def display_reactivity(frame):
    x, y, _, _ = display_coordinates(frame.hdr.shape[-2:], frame.output_size, frame.jitter)
    return at(frame.reactive, (x + 0.5).floor(), (y + 0.5).floor())


def temporal_error(frame, current, target, previous, valid):
    if frame.reset or previous is None or previous[0].shape != current.shape:
        return current.new_zeros(())
    height, width = frame.output_size
    x, y, _, _ = display_coordinates(frame.hdr.shape[-2:], frame.output_size, frame.jitter)
    near_x, near_y = (x + 0.5).floor(), (y + 0.5).floor()
    motion = at(frame.motion, near_x, near_y)
    mask = at(valid, near_x, near_y)
    dx, dy = grid(frame.output_size)
    px = dx - motion[0, 0] * width
    py = dy - motion[0, 1] * height
    mask = mask & (px >= -0.5) & (px < width - 0.5) & (py >= -0.5) & (py < height - 0.5)
    previous_error = sample_bilinear(previous[0].log1p() - previous[1].log1p(), px, py)
    return masked_mean(charbonnier(current.log1p() - target.log1p() - previous_error), mask)


def losses(sample, result, previous):
    prediction, target = result["display"], sample.target
    reactive = display_reactivity(sample.frame)
    delta = prediction.log1p() - target.log1p()
    robust = (charbonnier(delta) * (1 + 2 * reactive)).mean()
    energy = ((prediction - target).abs() / (1 + target)).mean()
    edge = edge_error(delta)
    temporal = temporal_error(sample.frame, prediction, target, previous, result["historyValid"])
    total = robust + 0.12 * energy + 0.08 * edge + 0.1 * temporal
    parts = {"loss": total, "logHdr": robust, "relativeEnergy": energy, "logEdge": edge,
             "warpedChange": temporal, "reactiveLogHdr": masked_mean(delta.abs(), reactive > 0),
             "linearRmse": (prediction - target).square().mean().sqrt()}
    if sample.area_target is not None:
        area = area_loss(result["areaVisibility"], sample.area_target, sample.frame.area_pairs)
        parts["areaMse"] = area
        total = total + 0.4 * area
    if sample.reflection_target is not None:
        mask = sample.frame.reflection[:, 3:4] > 0.5
        refl = masked_mean(charbonnier(result["reflection"].log1p()
                                      - sample.reflection_target.log1p()), mask)
        parts["reflectionLogHdr"] = refl
        total = total + 0.4 * refl
    parts["loss"] = total
    return parts


def fixed_display(hdr):
    mapped = hdr / (1 + hdr)
    return torch.where(mapped <= 0.0031308, 12.92 * mapped,
                       1.055 * mapped.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def display_metrics(prediction, target):
    prediction, target = fixed_display(prediction), fixed_display(target)
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    coordinates = torch.arange(-5, 6, dtype=torch.float32)
    gaussian = (-0.5 * (coordinates / 1.5).square()).softmax(dim=0)
    row = gaussian.reshape(1, 1, 1, 11).expand(3, 1, 1, 11).contiguous()
    column = row.transpose(-1, -2).contiguous()

    def filtered(image):
        horizontal = F.conv2d(F.pad(image, (5, 5, 0, 0), mode="replicate"), row, groups=3)
        return F.conv2d(F.pad(horizontal, (0, 0, 5, 5), mode="replicate"), column, groups=3)

    left, right = filtered(prediction), filtered(target)
    covariance = filtered(prediction * target) - left * right
    variance = (filtered(prediction.square()) - left.square()).clamp_min(0)
    variance += (filtered(target.square()) - right.square()).clamp_min(0)
    score = ((2 * left * right + 0.01**2) * (2 * covariance + 0.03**2)
             / ((left.square() + right.square() + 0.01**2) * (variance + 0.03**2)))
    return float(-10 * mse.log10()), float(score.clamp(-1, 1).mean())


def spatial_baseline(frame):
    x, y, _, _ = display_coordinates(frame.hdr.shape[-2:], frame.output_size, frame.jitter)
    return sample_bilinear(frame.display_source, x, y)


@torch.no_grad()
def evaluate(model, sequences):
    metrics = []
    for sequence in sequences:
        previous, previous_display, previous_spatial = None, None, None
        frames = []
        for sample in sequence.samples:
            result, previous = model(sample.frame, previous)
            values = {k: float(v) for k, v in losses(sample, result, previous_display).items()}
            previous_display = (result["display"], sample.target)
            spatial = spatial_baseline(sample.frame)
            values["spatialLinearRmse"] = float((spatial - sample.target).square().mean().sqrt())
            values["spatialLogMae"] = float((spatial.log1p() - sample.target.log1p()).abs().mean())
            values["logMae"] = float((result["display"].log1p() - sample.target.log1p()).abs().mean())
            values["displayPsnr"], values["displaySsim"] = display_metrics(result["display"], sample.target)
            values["spatialDisplayPsnr"], values["spatialDisplaySsim"] = display_metrics(spatial, sample.target)
            values["spatialWarpedChange"] = float(temporal_error(
                sample.frame, spatial, sample.target, previous_spatial, result["historyValid"]))
            previous_spatial = (spatial, sample.target)
            if sample.analytic is not None:
                values["analyticLinearRmse"] = float((sample.analytic - sample.target).square().mean().sqrt())
                values["analyticDisplayPsnr"], values["analyticDisplaySsim"] = display_metrics(sample.analytic, sample.target)
            values["historyValidFraction"] = float(result["historyValid"].float().mean())
            frames.append({"id": sample.id, **values})
        keys = sorted(set().union(*(frame.keys() for frame in frames)) - {"id"})
        mean = {k: sum(f[k] for f in frames if k in f) / sum(k in f for f in frames) for k in keys}
        metrics.append({"id": sequence.id, "groupId": sequence.group, "mean": mean, "frames": frames})
    return metrics


def area_diagnostic(model, records):
    errors = []
    for identity, frame, target in records:
        encoded, warped, gate, _ = model.encode(frame, None)
        estimate = model.visibility(encoded, warped * gate, frame.area_pairs)
        errors.append((identity, area_loss(estimate, target, frame.area_pairs)))
    return errors


def train(capture, output, steps, unroll, learning_rate, seed, area_capture):
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    model = seeded(seed)
    sequences, source = load_capture(capture)
    fit = [sequence for sequence in sequences if sequence.split == "train"]
    test = [sequence for sequence in sequences if sequence.split == "held-out"]
    if not fit:
        raise ValueError("training requires explicitly labeled train sequences")
    area_records, area_source = load_area_capture(area_capture) if area_capture else ([], None)
    provenance = {
        "gallery": source, "areaPilot": area_source,
        "trainFrameIds": [sample.id for sequence in fit for sample in sequence.samples],
        "heldOutFrameIds": [sample.id for sequence in test for sample in sequence.samples],
        "trainGroupIds": sorted({sequence.group for sequence in fit}),
        "trainFingerprints": [list(sample.fingerprint) for sequence in fit for sample in sequence.samples],
        "areaTrainFrameIds": [identity for identity, _, _ in area_records],
    }
    trained = ["display"]
    if area_records or any(s.area_target is not None for seq in fit for s in seq.samples):
        trained.append("area")
    if any(s.reflection_target is not None for seq in fit for s in seq.samples):
        trained.append("reflection")

    start_weights = sha(export_weights(model))
    initial = evaluate(model, fit)
    with torch.no_grad():
        area_before = {name: float(value) for name, value in area_diagnostic(model, area_records)}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for step in range(steps):
        sequence = fit[step % len(fit)]
        count = min(unroll, len(sequence.samples))
        start = (step // len(fit)) % (len(sequence.samples) - count + 1)
        previous, previous_display = None, None
        # A short burn-in uses only earlier, same-clip frames.
        with torch.no_grad():
            for sample in sequence.samples[max(0, start - 2):start]:
                output_frame, previous = model(sample.frame, previous)
                previous_display = (output_frame["display"], sample.target)

        total = 0.0
        for sample in sequence.samples[start:start + count]:
            output_frame, previous = model(sample.frame, previous)
            total = total + losses(sample, output_frame, previous_display)["loss"] / count
            previous_display = (output_frame["display"], sample.target)
        if area_records:
            _, frame, target = area_records[step % len(area_records)]
            total = total + 0.4 * area_diagnostic(model, [("ticket-10", frame, target)])[0][1]
        if not total.isfinite():
            raise ValueError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0:
            print(f"step {step + 1}/{steps} objective {float(total.detach()):.8f}", flush=True)

    manifest, reloaded = write_package(output, model, kind="windows-capture-fit", seed=seed,
                                       trained_heads=trained, provenance=provenance)
    final = evaluate(reloaded, fit)
    # Held-out labels are used only after training and export.
    held_out = evaluate(reloaded, test)
    with torch.no_grad():
        area_after = {name: float(value) for name, value in area_diagnostic(reloaded, area_records)}
        reload_peak = 0.0
        for sequence in fit + test:
            old, new = None, None
            for sample in sequence.samples:
                before, old = model(sample.frame, old)
                after, new = reloaded(sample.frame, new)
                reload_peak = max(reload_peak, max(float((before[k].float() - after[k].float()).abs().max())
                                                    for k in before))
    train_start = sum(seq["mean"]["loss"] for seq in initial) / len(initial)
    train_end = sum(seq["mean"]["loss"] for seq in final) / len(final)
    report = {
        "format": "raa.real-frame-fit-report/1", "steps": steps, "unroll": unroll, "seed": seed,
        "learningRate": learning_rate, "torchVersion": str(torch.__version__), "device": "cpu",
        "initialWeightsSha256": start_weights, "weightsSha256": manifest["weightsSha256"],
        "trainedHeads": trained, "provenance": provenance, "cpuReloadMaxAbs": reload_peak,
        "diagnosticDisplayTransform": "Reinhard x/(1+x), then sRGB",
        "overfit": {"initialLoss": train_start, "finalLoss": train_end,
                    "initial": initial, "final": final, "improved": train_end < train_start},
        "heldOut": {"usedForFit": False, "sequences": held_out,
                    "status": "scored" if test else "pending-independent-capture",
                    "qualification": "pilot trajectory only"},
        "ticket10AreaAuxiliary": {"initialTrainMse": area_before, "finalTrainMse": area_after},
        "shippingQualification": {"status": "unverified", "physicalLatencyAB": "not-run",
                                  "heldOutTitleCoverage": "not-established", "enabledByDefault": False},
    }
    (output / "fit-report.json").write_bytes(json_bytes(report))
    if not report["overfit"]["improved"] or reload_peak != 0.0:
        raise ValueError("real-frame diagnostic did not improve fit loss or failed CPU reload")
    return report


def score_held_out(capture, package, output):
    configure_cpu()
    manifest = json.loads((package / "manifest.json").read_bytes())
    if manifest["kind"] != "windows-capture-fit":
        raise ValueError("held-out scoring requires a fitted Windows capture package")
    weights = (package / "weights.fp32le").read_bytes()
    golden = (package / "golden.json").read_bytes()
    canonical = {k: v for k, v in manifest.items() if k != "packageSha256"}
    canonical = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    if manifest["goldenSha256"] != sha(golden) or manifest["packageSha256"] != sha(canonical + weights + golden):
        raise ValueError("real-frame package evidence hash mismatch")
    model = load_weights(manifest, weights)
    sequences, data = load_capture(capture)
    test = [sequence for sequence in sequences if sequence.split == "held-out"]
    if not test:
        raise ValueError("held-out scoring requires an explicitly test-labeled sequence")
    fit = manifest["provenance"]
    trained_groups = set(fit["trainGroupIds"])
    trained_ids = set(fit["trainFrameIds"])
    trained_fingerprints = {tuple(item) for item in fit["trainFingerprints"]}
    if any(seq.group in trained_groups for seq in test) or any(
        sample.id in trained_ids or sample.fingerprint in trained_fingerprints
        for seq in test for sample in seq.samples
    ):
        raise ValueError("held-out scoring overlaps the fitted groups, frames or plane fingerprints")
    report = {
        "format": "raa.real-frame-held-out-report/1", "weightsSha256": manifest["weightsSha256"],
        "packageSha256": manifest["packageSha256"], "heldOutCapture": data, "usedForFit": False,
        "diagnosticDisplayTransform": "Reinhard x/(1+x), then sRGB",
        "sequences": evaluate(model, test), "shippingQualification": "unverified",
    }
    with output.open("xb") as file:
        file.write(json_bytes(report))
    return report


def checkpoint(path):
    manifest = json.loads((path / "manifest.json").read_bytes())
    weights = (path / "weights.fp32le").read_bytes()
    if manifest["kind"] != "windows-capture-fit":
        raise ValueError("expected a trained real-frame checkpoint")
    golden_path = path / "golden.json"
    if golden_path.exists():
        golden = golden_path.read_bytes()
        canonical = {k: v for k, v in manifest.items() if k != "packageSha256"}
        canonical = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        if manifest["goldenSha256"] != sha(golden) or manifest["packageSha256"] != sha(canonical + weights + golden):
            raise ValueError("checkpoint package hash mismatch")
    return manifest, load_weights(manifest, weights)


def split_identity(sequences):
    return {
        "groupIds": sorted({seq.group for seq in sequences}),
        "replayHashes": sorted({seq.replay for seq in sequences if seq.replay}),
        "frameIds": sorted({sample.id for seq in sequences for sample in seq.samples}),
        "fingerprints": sorted({sample.fingerprint for seq in sequences for sample in seq.samples}),
    }


def require_separate(left, right):
    for name in left.keys() & right.keys():
        if name == "fingerprints":
            overlap = set(map(tuple, left[name])) & set(map(tuple, right[name]))
        else:
            overlap = set(left[name]) & set(right[name])
        if overlap:
            raise ValueError(f"trajectory split overlaps {name}: {next(iter(overlap))}")


def checkpoint_identity(manifest):
    provenance = manifest.get("provenance") or {}
    reflection_train = provenance.get("reflectionFit", {}).get("train", [])
    legacy = {
        "groupIds": list(provenance.get("trainGroupIds", [])) + [row["groupId"] for row in reflection_train],
        "frameIds": list(provenance.get("trainFrameIds", [])) + [frame for row in reflection_train for frame in row["frameIds"]],
        "fingerprints": provenance.get("trainFingerprints", []),
    }
    return [legacy, *provenance.get("fitAndSelectionSplits", {}).values()]


def paired_partition(paths, accepted):
    captures, sequences = [], []
    for path in paths:
        selected, receipt = load_capture(path, accepted)
        require_separate(split_identity(sequences), split_identity(selected))
        sequences.extend(selected)
        captures.append(receipt)
    return sequences, captures


def display_residual_change(frame, residual, prior, valid):
    if frame.reset or prior is None or prior.shape != residual.shape:
        return None
    height, width = frame.output_size
    x, y, _, _ = display_coordinates(frame.hdr.shape[-2:], frame.output_size, frame.jitter)
    near_x, near_y = (x + 0.5).floor(), (y + 0.5).floor()
    motion = at(frame.motion, near_x, near_y)
    dx, dy = grid(frame.output_size)
    px, py = dx - motion[0, 0] * width, dy - motion[0, 1] * height
    mask = at(valid, near_x, near_y)
    mask = mask & (px >= -0.5) & (px < width - 0.5) & (py >= -0.5) & (py < height - 0.5)
    return residual - sample_bilinear(prior, px, py), mask


def reflection_residual_change(frame, residual, prior, valid):
    if frame.reset or prior is None or prior[0].shape != residual.shape:
        return None
    previous, previous_guides, jitter = prior
    h, w = frame.hdr.shape[-2:]
    x, y = grid((h, w))
    px = x - frame.motion[0, 0] * w + jitter[0] - frame.jitter[0]
    py = y - frame.motion[0, 1] * h + jitter[1] - frame.jitter[1]
    old = at(previous_guides, (px + 0.5).floor(), (py + 0.5).floor())
    current = frame.reflection[:, 3:5]
    same_class = (old[:, :1] > 0.5) & (old[:, 1:2] == current[:, 1:2])
    mask = valid & same_class & (current[:, :1] > 0.5) & (frame.reactive < 0.5)
    return residual - sample_bilinear(previous, px, py), mask


def weighted_area(prediction, target, pairs):
    valid = pairs[:, 1:2] > 1e-8
    penumbra = (target > 0.02) & (target < 0.98)
    return masked_mean((prediction - target).square() * (1 + penumbra), valid)


def joint_loss(sample, result, previous_display, previous_reflection, target_mix):
    frame, prediction = sample.frame, result["display"]
    target = sample.raster_target if sample.raster_target is not None else sample.target
    delta = prediction.log1p() - target.log1p()
    display_delta = fixed_display(prediction) - fixed_display(target)
    reactive = display_reactivity(frame)
    log_hdr = (charbonnier(delta) * (1 + reactive)).mean()
    display = (charbonnier(display_delta) * (1 + reactive)).mean()
    energy = ((prediction - target).abs() / (1 + target)).mean()
    change = display_residual_change(frame, delta, previous_display, result["historyValid"])
    temporal = masked_mean(charbonnier(change[0]), change[1]) if change is not None else 0.0
    total = 0.50 * log_hdr + 0.65 * display + 0.10 * energy + 0.24 * edge_error(display_delta) + 0.40 * temporal
    if sample.raster_target is not None and target_mix > 0:
        path_delta = prediction.log1p() - sample.target.log1p()
        total = total + target_mix * charbonnier(path_delta).mean()
    if sample.area_target is not None:
        total = total + 0.4 * weighted_area(result["areaVisibility"], sample.area_target, frame.area_pairs)
    ref_state = None
    if sample.reflection_target is not None:
        reflect = result["reflection"]
        residual = reflect.log1p() - sample.reflection_target.log1p()
        active = frame.reflection[:, 3:4] > 0.5
        relative = (reflect - sample.reflection_target).abs() / (1 + sample.reflection_target)
        reflection = masked_mean(charbonnier(residual) + 0.16 * relative, active)
        change = reflection_residual_change(frame, residual, previous_reflection, result["historyValid"])
        if change is not None:
            reflection = reflection + 0.30 * masked_mean(charbonnier(change[0]), change[1])
        total = total + 0.60 * reflection
        ref_state = (residual, frame.reflection[:, 3:5], frame.jitter)
    return total, delta, ref_state


def mean_rows(rows):
    keys = set().union(*(row.keys() for row in rows))
    return {k: sum(row[k] for row in rows if k in row) / sum(k in row for row in rows)
            for k in sorted(keys)}


@torch.no_grad()
def joint_metrics(models, sequences):
    all_sequences = []
    for sequence in sequences:
        histories = dict.fromkeys(models)
        display_errors, reflection_errors = {}, {}
        rows = []
        for sample in sequence.samples:
            f = sample.frame
            predictions = {}
            for name, model in models.items():
                predictions[name], histories[name] = model(f, histories[name])
            valid = next(iter(predictions.values()))["historyValid"]
            raw_area = visibility_ratio(f.area_pairs[:, :1], f.area_pairs[:, 1:2]) if f.area_pairs is not None else None
            predictions["bilinear/raw"] = {"display": spatial_baseline(f), "areaVisibility": raw_area,
                                            "reflection": f.reflection[:, :3] if f.reflection is not None else None}
            frame_metrics = {}
            for name, result in predictions.items():
                row = {}
                for label, teacher in (("path", sample.target), ("raster", sample.raster_target)):
                    if teacher is None:
                        continue
                    prediction = result["display"]
                    linear = prediction - teacher
                    delta = prediction.log1p() - teacher.log1p()
                    psnr, ssim = display_metrics(prediction, teacher)
                    row.update({f"{label}Psnr": psnr, f"{label}Ssim": ssim,
                                f"{label}HdrRmse": float(linear.square().mean().sqrt()),
                                f"{label}LogMae": float(delta.abs().mean()),
                                f"{label}EnergyBias": float(linear.sum() / teacher.sum().clamp_min(1e-8))})
                    key = (name, label)
                    change = display_residual_change(f, delta, display_errors.get(key), valid)
                    if change is not None and change[1].any():
                        row[f"{label}WarpedLogChangeMae"] = float(masked_mean(change[0].abs(), change[1]))
                    display_errors[key] = delta
                    if sample.event:
                        row[f"{label}EventLogMae"] = float(delta.abs().mean())
                    if sample.disocclusion is not None and sample.disocclusion.any():
                        x, y, _, _ = display_coordinates(f.hdr.shape[-2:], f.output_size, f.jitter)
                        mask = at(sample.disocclusion, (x + 0.5).floor(), (y + 0.5).floor())
                        row[f"{label}DisocclusionLogMae"] = float(masked_mean(delta.abs(), mask))
                if sample.reflection_target is not None:
                    active = f.reflection[:, 3:4] > 0.5
                    if active.any():
                        reflect = result["reflection"]
                        delta = reflect.log1p() - sample.reflection_target.log1p()
                        row["reflectionLogMae"] = float(masked_mean(delta.abs(), active))
                        row["reflectionHdrRmse"] = float(masked_mean((reflect - sample.reflection_target).square(), active).sqrt())
                        change = reflection_residual_change(f, delta, reflection_errors.get(name), valid)
                        if change is not None and change[1].any():
                            row["reflectionWarpedLogChangeMae"] = float(masked_mean(change[0].abs(), change[1]))
                        reflection_errors[name] = (delta, f.reflection[:, 3:5], f.jitter)
                else:
                    reflection_errors.pop(name, None)
                if sample.area_target is not None:
                    pairs, area = f.area_pairs, result["areaVisibility"]
                    active = pairs[:, 1:2] > 1e-8
                    penumbra = active & (sample.area_target > 0.02) & (sample.area_target < 0.98)
                    row["areaVisibilityRmse"] = float(area_loss(area, sample.area_target, pairs).sqrt())
                    if penumbra.any():
                        row["areaPenumbraMae"] = float(masked_mean((area - sample.area_target).abs(), penumbra))
                frame_metrics[name] = row
            rows.append({"id": sample.id, "metrics": frame_metrics})
        means = {name: mean_rows([row["metrics"][name] for row in rows]) for name in predictions}
        all_sequences.append({"id": sequence.id, "groupId": sequence.group, "mean": means, "frames": rows})
    return {
        "mean": {name: mean_rows([seq["mean"][name] for seq in all_sequences])
                 for name in (*models, "bilinear/raw")},
        "sequences": all_sequences,
    }


def development_score(report, reference):
    scores = report["mean"]
    row, raw = scores["candidate"], scores["bilinear/raw"]
    psnr = f"{reference}Psnr"
    ratios = [10 ** ((raw[psnr] - row[psnr]) / 20)]
    terms = [(f"{reference}HdrRmse", 1.0), (f"{reference}WarpedLogChangeMae", 1.25),
             ("reflectionHdrRmse", 1.0), ("reflectionLogMae", 1.0), ("areaVisibilityRmse", 0.5)]
    weights = [1.0]
    for key, weight in terms:
        if key in raw and raw[key] > 0:
            ratios.append(row[key] / raw[key])
            weights.append(weight)
    ssim = f"{reference}Ssim"
    ratios.append((1 - row[ssim]) / max(1 - raw[ssim], 1e-6))
    weights.append(1.0)
    return sum(ratio * weight for ratio, weight in zip(ratios, weights)) / sum(weights)


def fit_joint(options):
    configure_cpu()
    root = options.output
    if root.exists():
        raise ValueError(f"output already exists: {root}")
    fit, fit_sources = paired_partition(options.train, {"train"})
    development, dev_sources = paired_partition(options.development, {"development"})
    if any(not seq.replay for seq in fit + development):
        raise ValueError("joint training and development require recorded trajectory replay hashes")
    identities = {"train": split_identity(fit), "development": split_identity(development)}
    require_separate(*identities.values())
    original, model = checkpoint(options.base)
    _, existing = checkpoint(options.base)
    for used in checkpoint_identity(original):
        require_separate(used, identities["development"])
    auxiliary, area_source = load_area_capture(options.area_capture) if options.area_capture else ([], None)
    if not auxiliary and not any(sample.area_target is not None for seq in fit for sample in seq.samples):
        raise ValueError("joint fit requires area visibility supervision")
    if not any(sample.reflection_target is not None and (sample.frame.reflection[:, 3:4] > 0.5).any()
               for seq in fit for sample in seq.samples):
        raise ValueError("joint fit requires supported reflection radiance supervision")
    if options.display_init == "bilinear":
        torch.nn.init.zeros_(model.sr.weight)
        torch.nn.init.zeros_(model.sr.bias)
    initial_hash = sha(export_weights(model))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(options.seed)
    reference = "raster" if all(s.raster_target is not None for seq in development for s in seq.samples) else "path"
    trace, best = [], None
    for step in range(1, options.steps + 1):
        sequence = fit[(step - 1) % len(fit)]
        count = min(options.unroll, len(sequence.samples))
        start = int(torch.randint(len(sequence.samples) - count + 1, (), generator=generator))
        previous, previous_display, previous_reflection = None, None, None
        with torch.no_grad():
            for sample in sequence.samples[max(0, start - 4):start]:
                result, previous = model(sample.frame, previous)
                _, previous_display, previous_reflection = joint_loss(
                    sample, result, previous_display, previous_reflection, options.path_weight)
        total = torch.zeros(())
        for sample in sequence.samples[start:start + count]:
            result, previous = model(sample.frame, previous)
            loss, previous_display, previous_reflection = joint_loss(
                sample, result, previous_display, previous_reflection, options.path_weight)
            total = total + loss / count
        if auxiliary:
            _, frame, target = auxiliary[(step - 1) % len(auxiliary)]
            encoded, warped, gate, _ = model.encode(frame, None)
            estimate = model.visibility(encoded, warped * gate, frame.area_pairs)
            total = total + 0.4 * weighted_area(estimate, target, frame.area_pairs)
        if not total.isfinite():
            raise ValueError(f"nonfinite joint loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % options.validate_every == 0 or step == options.steps:
            report = joint_metrics({"candidate": model, "existing": existing}, development)
            score = development_score(report, reference)
            trace.append({"step": step, "trainWindowLoss": float(total.detach()), "selectionScore": score,
                          "development": report["mean"]})
            print(f"step {step}/{options.steps} train {float(total.detach()):.7f} development {score:.7f}", flush=True)
            if best is None or score < best[0]:
                best = (score, step, export_weights(model), report)
    score, selected_step, weights, report = best
    selected_model = load_weights({**original, "weightsSha256": sha(weights)}, weights)
    provenance = {
        "basePackageSha256": original["packageSha256"], "baseWeightsSha256": original["weightsSha256"],
        "fitAndSelectionSplits": {**identities, **{f"upstream{i}": used for i, used in enumerate(checkpoint_identity(original))}},
        "trainGroupIds": identities["train"]["groupIds"], "trainFrameIds": identities["train"]["frameIds"],
        "trainFingerprints": identities["train"]["fingerprints"],
        "trainCaptures": fit_sources, "developmentCaptures": dev_sources, "areaPilot": area_source,
        "areaTrainFrameIds": [identity for identity, _, _ in auxiliary],
        "selectedStep": selected_step,
        "decisionSplit": "development", "finalHoldoutUsed": False,
        "displayInitialization": options.display_init, "pathAuxiliaryWeight": options.path_weight,
        "learningRate": options.learning_rate, "unroll": options.unroll, "validatedSteps": options.steps,
        "lossProfile": "matched-raster-log-display-edge-energy-warped-change-area-reflection-v1",
        "sourceRouting": "forward HDR and raw incident for recurrence; same-frame SSR composite for display",
    }
    manifest, reloaded = write_package(root, selected_model, kind="windows-capture-fit", seed=options.seed,
                                       trained_heads=["display", "area", "reflection"], provenance=provenance)
    if joint_metrics({"candidate": reloaded, "existing": existing}, development) != report:
        raise ValueError("frozen package changed the development rollout")
    training_metrics = joint_metrics({"candidate": reloaded, "existing": existing}, fit)
    original_params = existing.state_dict()
    parameter_shift = {key: float((weight - original_params[key]).abs().max())
                       for key, weight in reloaded.state_dict().items()}
    exports = {
        "format": "raa.joint-real-frame-fit/1", "torchVersion": str(torch.__version__), "device": "cpu",
        "weightsSha256": manifest["weightsSha256"], "packageSha256": manifest["packageSha256"],
        "initialWeightsSha256": initial_hash, "selectionTeacher": reference,
        "selectedStep": selected_step, "selectionScore": score, "provenance": provenance,
        "training": training_metrics, "development": report, "checkpoints": trace,
        "parameterMaxAbsChange": parameter_shift,
    }
    (root / "joint-fit-report.json").write_bytes(json_bytes(exports))
    return exports


def compare_joint(options):
    configure_cpu()
    candidate, model = checkpoint(options.candidate)
    baseline, existing = checkpoint(options.base)
    if (not (options.candidate / "golden.json").is_file()
            or candidate.get("provenance", {}).get("decisionSplit") != "development"):
        raise ValueError("final scoring requires a frozen, development-selected candidate")
    final, receipts = paired_partition(options.final, {"held-out"})
    if any(not seq.replay for seq in final):
        raise ValueError("final trajectories require recorded replay hashes")
    identity = split_identity(final)
    for package in (candidate, baseline):
        for used in checkpoint_identity(package):
            require_separate(used, identity)
    result = {
        "format": "raa.joint-real-frame-held-out/1", "selectionUsedFinal": False,
        "candidateWeightsSha256": candidate["weightsSha256"], "candidatePackageSha256": candidate["packageSha256"],
        "existingWeightsSha256": baseline["weightsSha256"], "existingPackageSha256": baseline["packageSha256"],
        "heldOut": receipts, "heldOutIdentity": identity,
        "displayTransform": "fixed unit exposure Reinhard then sRGB",
        "temporal": "MAE of current log reconstruction residual minus the motion-warped previous residual",
        "aggregation": "within-trajectory mean over eligible frames, then equal mean over trajectories",
        "shippingQualification": "unverified",
        **joint_metrics({"candidate": model, "existing": existing}, final),
    }
    with options.output.open("xb") as file:
        file.write(json_bytes(result))
    return result


def main():
    app = argparse.ArgumentParser(description=__doc__)
    action = app.add_subparsers(dest="action", required=True)
    parity = action.add_parser("seeded")
    parity.add_argument("output", type=Path)
    parity.add_argument("--seed", type=int, default=0)
    fit = action.add_parser("fit")
    fit.add_argument("capture", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--area-capture", type=Path)
    fit.add_argument("--steps", type=int, default=80)
    fit.add_argument("--unroll", type=int, default=4)
    fit.add_argument("--learning-rate", type=float, default=0.003)
    fit.add_argument("--seed", type=int, default=0)
    score = action.add_parser("score-held-out")
    score.add_argument("capture", type=Path)
    score.add_argument("package", type=Path)
    score.add_argument("output", type=Path)
    joint = action.add_parser("fit-joint")
    joint.add_argument("output", type=Path)
    joint.add_argument("--train", nargs="+", type=Path, required=True)
    joint.add_argument("--development", nargs="+", type=Path, required=True)
    joint.add_argument("--base", type=Path, required=True)
    joint.add_argument("--area-capture", type=Path)
    joint.add_argument("--steps", type=int, default=80)
    joint.add_argument("--unroll", type=int, default=6)
    joint.add_argument("--validate-every", type=int, default=20)
    joint.add_argument("--learning-rate", type=float, default=0.001)
    joint.add_argument("--seed", type=int, default=0)
    joint.add_argument("--display-init", choices=("checkpoint", "bilinear"), default="bilinear")
    joint.add_argument("--path-weight", type=float, default=0.15)
    compare = action.add_parser("compare-joint")
    compare.add_argument("output", type=Path)
    compare.add_argument("--final", nargs="+", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--base", type=Path, required=True)
    options = app.parse_args()
    if options.action == "fit-joint":
        if (min(options.steps, options.validate_every) < 1 or options.unroll < 2
                or not math.isfinite(options.learning_rate) or options.learning_rate <= 0
                or not math.isfinite(options.path_weight) or options.path_weight < 0):
            app.error("invalid joint optimization settings")
        result = fit_joint(options)
        print(f"frozen step {result['selectedStep']} weights {result['weightsSha256']}")
    elif options.action == "compare-joint":
        print(json.dumps(compare_joint(options)["mean"], indent=2))
    elif options.action == "seeded":
        manifest, _ = write_package(options.output, seeded(options.seed),
                                    kind="seeded-parity-only", seed=options.seed)
        print(manifest["weightsSha256"])
    elif options.action == "score-held-out":
        report = score_held_out(options.capture, options.package, options.output)
        for sequence in report["sequences"]:
            print(f"{sequence['id']}: held-out loss {sequence['mean']['loss']:.8f}")
    else:
        if options.steps < 1 or options.unroll < 2 or not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
            app.error("fit requires positive steps, an unroll of at least two and a positive finite learning rate")
        report = train(options.capture, options.output, options.steps, options.unroll,
                       options.learning_rate, options.seed, options.area_capture)
        print(f"train loss {report['overfit']['initialLoss']:.8f} → {report['overfit']['finalLoss']:.8f}")


if __name__ == "__main__":
    main()
