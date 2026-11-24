#!/usr/bin/env python3
"""Fit and export the causal real-frame FP32 ABI on lossless Windows captures."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

import torch

from real_frame_model import (
    Frame, at, display_coordinates, export_weights, grid, json_bytes, sample_bilinear,
    seeded, sha, write_package,
)


CAPTURE_FORMAT = "raa.gallery-temporal-sequence/1"


@dataclass
class Sample:
    id: str
    frame: Frame
    target: torch.Tensor
    area_target: torch.Tensor | None
    reflection_target: torch.Tensor | None
    analytic: torch.Tensor | None


@dataclass
class Sequence:
    id: str
    group: str
    split: str
    samples: list


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


def load_capture(path):
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
        split = {"train": "train", "test": "held-out", "held-out": "held-out"}.get(sequence["split"])
        if not isinstance(seq_id, str) or not seq_id or seq_id in sequence_ids or not group or split is None:
            raise ValueError("sequence IDs and split assignments must be explicit and unique")
        if occupied_groups.setdefault(group, split) != split:
            raise ValueError(f"split leakage for capture group {group}")
        sequence_ids.add(seq_id)
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
            analytic = read_plane(path.parent, p["analyticDisplayHdr"], (3,), target.shape[-2:]) if "analyticDisplayHdr" in p else None
            for image, label in ((hdr, "HDR"), (target, "teacher HDR"), (depth, "metric depth")):
                require_range(image, label, 0.0)
            for image, label in ((reactive, "reactive"), (history_valid, "history validity"), (guide[:, 4:5], "roughness")):
                require_range(image, label, 0.0, 1.0)
            if analytic is not None:
                require_range(analytic, "analytic HDR", 0.0)
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
                          jitter, target.shape[-2:], reset, areas, reflections)
            samples.append(Sample(identity, frame, target, area_target, reflection_target, analytic))
        if len(samples) < 2:
            raise ValueError(f"sequence {seq_id} needs at least two real frames")
        sequences.append(Sequence(seq_id, group, split, samples))
    if {sequence.split for sequence in sequences} != {"train", "held-out"}:
        raise ValueError("independent train and held-out sequences are required")
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


def spatial_baseline(frame):
    x, y, _, _ = display_coordinates(frame.hdr.shape[-2:], frame.output_size, frame.jitter)
    return sample_bilinear(frame.hdr, x, y)


@torch.no_grad()
def evaluate(model, sequences):
    metrics = []
    for sequence in sequences:
        previous, previous_display = None, None
        frames = []
        for sample in sequence.samples:
            result, previous = model(sample.frame, previous)
            values = {k: float(v) for k, v in losses(sample, result, previous_display).items()}
            previous_display = (result["display"], sample.target)
            spatial = spatial_baseline(sample.frame)
            values["spatialLinearRmse"] = float((spatial - sample.target).square().mean().sqrt())
            values["spatialLogMae"] = float((spatial.log1p() - sample.target.log1p()).abs().mean())
            if sample.analytic is not None:
                values["analyticLinearRmse"] = float((sample.analytic - sample.target).square().mean().sqrt())
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
    area_records, area_source = load_area_capture(area_capture) if area_capture else ([], None)
    provenance = {
        "gallery": source, "areaPilot": area_source,
        "trainFrameIds": [sample.id for sequence in fit for sample in sequence.samples],
        "heldOutFrameIds": [sample.id for sequence in test for sample in sequence.samples],
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
            print(f"step {step + 1}/{steps} objective {float(total):.8f}", flush=True)

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
        "overfit": {"initialLoss": train_start, "finalLoss": train_end,
                    "initial": initial, "final": final, "improved": train_end < train_start},
        "heldOut": {"independentFit": False, "sequences": held_out,
                    "qualification": "pilot trajectory only"},
        "ticket10AreaAuxiliary": {"initialTrainMse": area_before, "finalTrainMse": area_after},
        "shippingQualification": {"status": "unverified", "physicalLatencyAB": "not-run",
                                  "heldOutTitleCoverage": "not-established", "enabledByDefault": False},
    }
    (output / "fit-report.json").write_bytes(json_bytes(report))
    if not report["overfit"]["improved"] or reload_peak != 0.0:
        raise ValueError("real-frame diagnostic did not improve fit loss or failed CPU reload")
    return report


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
    options = app.parse_args()
    if options.action == "seeded":
        manifest, _ = write_package(options.output, seeded(options.seed),
                                    kind="seeded-parity-only", seed=options.seed)
        print(manifest["weightsSha256"])
    else:
        if options.steps < 1 or options.unroll < 2 or not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
            app.error("fit requires positive steps, an unroll of at least two and a positive finite learning rate")
        report = train(options.capture, options.output, options.steps, options.unroll,
                       options.learning_rate, options.seed, options.area_capture)
        print(f"train loss {report['overfit']['initialLoss']:.8f} → {report['overfit']['finalLoss']:.8f}")


if __name__ == "__main__":
    main()
