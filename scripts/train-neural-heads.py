#!/usr/bin/env python3
"""Fit the reflection and generated-frame heads with independent Windows targets."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

import generated_frame_model as generation
from neural_capture import frame_id, read_plane, require_range, require_windows
import real_frame_model as real


@dataclass
class Example:
    id: str
    inputs: object
    target: torch.Tensor
    label_sha: str
    input_sha: str


@dataclass
class SplitGroup:
    id: str
    group: str
    split: str
    examples: list


def read_capture(path, signal, read):
    data = path.read_bytes()
    manifest = json.loads(data)
    native_reflection = (signal == "raa.reflection-teacher/1"
                         and manifest["format"] == "raa.gallery-temporal-sequence/1")
    if manifest["format"] != signal and not native_reflection:
        raise ValueError(f"expected {signal}")
    if native_reflection:
        reflection = manifest.get("optionalHeads", {}).get("reflection", {})
        if (not isinstance(reflection, dict)
                or reflection.get("supplier") != "typed SSR incident and hitMeta from the same real submission"
                or reflection.get("eligibility") != "class 1 with geometric validity and positive screen confidence"):
            raise ValueError("reflection capture must declare its typed, supported screen estimator")
    require_windows(manifest)
    if manifest.get("rowOrder", "top-down") != "top-down" or manifest.get("exposure", 1) != 1:
        raise ValueError("expected top-down scene-linear unit-exposure planes")
    captured = []
    set_ids = set()
    frame_ids = set()
    split_by_group = {}
    fingerprints = {}
    for sequence in manifest["sequences"]:
        identity = frame_id(sequence)
        group = sequence["groupId"]
        split = {"train": "train", "test": "held-out", "held-out": "held-out"}.get(sequence["split"])
        if (identity in set_ids or not isinstance(group, str) or not group or split is None
                or split_by_group.setdefault(group, split) != split):
            raise ValueError(f"invalid split or repeated sequence {identity}")
        set_ids.add(identity)
        frames = []
        for index, frame in enumerate(sequence["frames"]):
            if native_reflection:
                clock = frame["frameId"]
                if index and not frame.get("reset") and (
                        clock != previous_clock + 1 or frame["viewId"] != previous_view
                        or not frame["timeMs"] > previous_time):
                    raise ValueError("reflection recurrence requires consecutive real frames or an explicit reset")
                previous_clock, previous_view, previous_time = clock, frame["viewId"], frame["timeMs"]
                sample = read(path.parent, frame, native_screen=True)
            else:
                sample = read(path.parent, frame)
            if sample.id in frame_ids:
                raise ValueError(f"repeated training frame {sample.id}")
            frame_ids.add(sample.id)
            fingerprint = (sample.input_sha, sample.label_sha)
            if fingerprints.setdefault(fingerprint, split) != split:
                raise ValueError(f"reused input and teacher across splits at {sample.id}")
            frames.append(sample)
        if not frames:
            raise ValueError(f"missing frames at {identity}")
        captured.append(SplitGroup(identity, group, split, frames))
    if {group.split for group in captured} != {"train", "held-out"}:
        raise ValueError("require disjoint train and held-out captures")
    capture = {"sha256": real.sha(data), "source": manifest.get("source", manifest.get("facts"))}
    if native_reflection:
        capture["signal"] = manifest["optionalHeads"]["reflection"]
    return captured, capture


def reflection_example(root, frame, native_screen=False):
    p = frame["planes"]
    beauty = read_plane(root, p["hdr"], (3,))
    size = beauty.shape[-2:]

    def plane(name, channels):
        return read_plane(root, p[name], channels, size)

    g = plane("guide", (12,))
    normal = torch.cat((g[:, :3], g[:, 4:5]), dim=1)
    raw = plane("reflection", (6,))
    target = plane("reflectionTarget", (3,))
    depth = plane("depth", (1,))
    motion = plane("motion", (3, 4))[:, :3]
    jitter = tuple(frame.get("jitterPx", g[0, 9:11, 0, 0].tolist()))
    if len(jitter) != 2 or not all(math.isfinite(x) for x in jitter):
        raise ValueError("invalid reflection jitter")
    for value, name in ((beauty, "scene HDR"), (raw, "incident capture"), (target, "incident teacher"),
                        (depth, "metric depth")):
        require_range(value, name, 0)
    require_range(raw[:, 3:4], "reflection validity", 0, 1)
    require_range(raw[:, 4:5], "reflection source class", 0, 4)
    teacher = frame["teacher"]
    batches = teacher.get("seedBatches", teacher.get("seeds"))
    input_seeds = frame["inputSeeds"]
    valid_seed = lambda value: type(value) is int and 0 <= value <= 2**32 - 1
    if (any(not valid_seed(seed) for seed in input_seeds)
            or (not input_seeds and not native_screen)
            or not isinstance(batches, list) or len(batches) != 2
            or len(batches[0]) != len(batches[1])):
        raise ValueError("reflection capture needs independent input and two balanced teacher batches")
    seed_set = set(input_seeds)
    for batch in batches:
        if (not batch or any(not valid_seed(seed) for seed in batch)
                or len(batch) != len(set(batch)) or seed_set.intersection(batch)):
            raise ValueError("overlapping or invalid reflection teacher seeds")
        seed_set.update(batch)
    convergence = teacher.get("reflection", teacher)
    if not 0 <= convergence["logRmse"] <= convergence["tolerance"]:
        raise ValueError("independent reflection teacher batches did not converge")
    support = raw[:, 3:4] > 0.5
    if not support.any() or (native_screen and (
            not (raw[:, 4:5][support] == 1).all()
            or int(support.sum()) != convergence["accepted"]
            or (target * (~support)).abs().max() != 0)):
        raise ValueError("reflection teacher validity differs from the screen support")
    initial = real.Frame(beauty, normal, depth, motion, g[:, 8:9], g[:, 7:8],
                         jitter, size, frame.get("reset", False), reflection=raw)
    return Example(frame_id(frame), initial, target, p["reflectionTarget"]["sha256"],
                   p["reflection"]["sha256"])


def finite_clock(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def audit_cutoff(frame):
    a = frame["sourceAvailableMs"]
    g = frame["guidesAvailableMs"]
    cutoff = frame["cutoffMs"]
    target = frame["targetRenderStartedMs"]
    if not all(finite_clock(x) for x in (a, g, cutoff, target)) or not max(a, g) <= cutoff < target:
        raise ValueError("generated-frame packet consumed data after the cutoff")
    source_id, target_id = frame["sourceRealFrameId"], frame["targetRealFrameId"]
    age = frame["sourceAgeRealFrames"]
    if (any(type(n) is not int for n in (source_id, target_id, age))
            or not 0 <= source_id < target_id or age != target_id - source_id):
        raise ValueError("generated-frame packet requires an earlier real source")
    snapshot = frame["acceptedTarget"]
    receipts = snapshot["inputs"]
    ids, times, confirmed = receipts["inputIds"], receipts["receivedAtMs"], receipts["confirmedMask"]
    if (receipts["cutoffMs"] != cutoff or snapshot["observedAtMs"] != cutoff
            or not ids or len(ids) != len(times) or len(ids) != len(confirmed)
            or snapshot.get("discontinuities")):
        raise ValueError("generated-frame packet has a missing cutoff, receipts or discontinuity guard")
    source = frame["sourceSnapshot"]
    if (source["target"]["tick"] >= snapshot["target"]["tick"]
            or source["revision"] != snapshot["revision"]):
        raise ValueError("generated-frame snapshots cross a revision or use a noncausal target")
    for identity, time, accepted in zip(ids, times, confirmed):
        if accepted:
            if not identity or not finite_clock(time) or time > cutoff:
                raise ValueError("generated-frame packet contains a post-cutoff input receipt")
        elif identity is not None or time is not None:
            raise ValueError("unconfirmed generated-frame input cannot carry an accepted receipt")
    return age


def replay_warp(packet, source_hdr, source_depth):
    """Compute the pre-learning visibility decision in prediction_warp.wgsl."""
    h, w = packet.depth.shape[-2:]
    x, y = real.grid((h, w))
    u = (x + 0.5) / w - packet.motion[0, 0]
    v = (y + 0.5) / h - packet.motion[0, 1]
    sx, sy = (u * w).floor(), (v * h).floor()
    expected = packet.motion[:, 3:4]
    depth = real.at(source_depth, sx, sy)
    color = real.at(source_hdr, sx, sy)[:, :3]
    valid = ((u >= 0) & (u < 1) & (v >= 0) & (v < 1) & (packet.motion[:, 2:3] < 0.5)
             & (packet.lighting == 0) & (packet.depth >= 0) & (packet.depth <= 1)
             & (expected >= 0) & (expected <= 1) & ((depth == 1) == (expected == 1))
             & ((packet.depth == 1) == (expected == 1)) & ((depth - expected).abs() <= 0.0005)
             & (color.abs() <= 65504).all(dim=1, keepdim=True))
    return torch.cat((torch.where(valid, color, 0.0), valid), dim=1)


def audit_warp(packet, source_hdr, source_depth, correction=None):
    expected = replay_warp(packet, source_hdr, source_depth)
    valid = expected[:, 3:4] > 0.5
    mismatch = int((packet.warp != expected).any(dim=1).sum())
    fresh = packet.fresh[:, 3:4] > 0.5
    redundant = int((fresh & valid).sum())
    if mismatch and correction is None:
        raise ValueError("retained HDR warp does not match admitted runtime guides")
    if not (valid | fresh).all() or (redundant and correction is None):
        raise ValueError("prediction training packet has incorrect or incomplete fresh repairs")
    if correction is not None:
        packet.warp = expected
        packet.fresh = torch.where(valid, 0.0, packet.fresh)
        correction.update(warpMismatchPixels=mismatch, excludedFreshPixels=redundant)


def generated_example(root, frame, correction=None):
    age = audit_cutoff(frame)
    p = frame["planes"]
    target = read_plane(root, p["label"], (3,))
    size = target.shape[-2:]

    def plane(name, channels):
        return read_plane(root, p[name], channels, size)

    source_hdr = plane("sourceHdr", (3, 4))
    source_depth = plane("sourceDepth", (1,))
    # The browser refresh provider writes RGBA16Float before learned refinement.
    fresh = plane("fresh", (4,)).half().float()
    packet = generation.Packet(plane("warp", (4,)), fresh, plane("targetDepth", (1,)),
                               plane("motion", (4,)), plane("lightingChanged", (1,)), age)
    for value, name in ((source_hdr, "source HDR"), (packet.warp, "warped HDR"),
                        (packet.fresh, "fresh HDR"), (target, "target HDR")):
        require_range(value, name, 0, 65504)
    require_range(packet.depth, "target depth", 0, 1)
    require_range(packet.lighting, "target light-change mask", 0, 1)
    audit_warp(packet, source_hdr, source_depth, correction)
    return Example(frame_id(frame), packet, target, p["label"]["sha256"], p["warp"]["sha256"])


def region_average(value, mask):
    return (value * mask).sum() / (mask.sum().clamp_min(1) * value.shape[1])


def objective(output, target, base, mask):
    delta = output.log1p() - target.log1p()
    base_error = (base.log1p() - target.log1p()).abs()
    importance = 1 + 3 * (base_error.amax(dim=1, keepdim=True) > 0.006)
    loss = region_average(importance * ((delta.square() + 1e-5).sqrt() - math.sqrt(1e-5)), mask)
    loss += 0.15 * region_average((output - target).abs() / (1 + target), mask)
    return loss


def frame_metrics(output, base, target, mask, exact=None):
    def values(rgb):
        difference = rgb - target
        return {
            "linearRmse": float(difference.square().mean().sqrt()),
            "activeLinearRmse": float(region_average(difference.square(), mask).sqrt()),
            "logMae": float((rgb.log1p() - target.log1p()).abs().mean()),
            "activeLogMae": float(region_average((rgb.log1p() - target.log1p()).abs(), mask)),
            "linearEnergyBias": float(difference.mean()),
        }
    result = {"learned": values(output), "analytic": values(base), "activePixels": int(mask.sum())}
    result["learnedDeltaMaxAbs"] = float((output - base).abs().max())
    if exact is not None:
        result["freshMaxAbs"] = float((torch.where(exact, output - base, 0.0)).abs().max())
    return result


def aggregate(rows):
    return {
        mode: {k: sum(frame[mode][k] for frame in rows) / len(rows) for k in rows[0][mode]}
        for mode in ("learned", "analytic")
    }


def measure(groups, forward):
    summaries = []
    for group in groups:
        frames = [{"id": item.id, **frame_metrics(*forward(item))} for item in group.examples]
        summaries.append({"id": group.id, "groupId": group.group, "mean": aggregate(frames),
                          "frames": frames})
    return summaries


def provenance(capture, groups):
    return {
        "capture": capture,
        "train": [{"id": g.id, "groupId": g.group, "frameIds": [x.id for x in g.examples]}
                  for g in groups if g.split == "train"],
        "heldOut": [{"id": g.id, "groupId": g.group, "frameIds": [x.id for x in g.examples]}
                    for g in groups if g.split == "held-out"],
    }


def frozen_package(path):
    manifest = json.loads((path / "manifest.json").read_bytes())
    weights = (path / "weights.fp32le").read_bytes()
    golden = (path / "golden.json").read_bytes()
    fields = {key: value for key, value in manifest.items() if key != "packageSha256"}
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    if manifest["goldenSha256"] != real.sha(golden) or manifest["packageSha256"] != real.sha(canonical + weights + golden):
        raise ValueError("base package hash does not match frozen assets")
    return manifest, weights


@torch.no_grad()
def reflection_features(model, groups):
    for group in groups:
        previous = None
        for item in group.examples:
            frame = item.inputs
            results, previous = model(frame, previous)
            r = frame.reflection
            mask = r[:, 3:4] > 0.5
            item.inputs = (results["state"].detach(), r[:, :3], mask)


def fit_reflection(args, groups, receipt):
    if args.base is None:
        raise ValueError("reflection fit requires a base real-frame package")
    manifest, weights = frozen_package(args.base)
    model = real.load_weights(manifest, weights)
    model.requires_grad_(False)
    model.reflection.requires_grad_(True)
    reflection_features(model, groups)

    def forward(item, selected=model):
        hidden, raw, valid = item.inputs
        prediction = torch.where(valid, real.residual(raw, selected.reflection(hidden), 0.5), 0)
        base = torch.where(valid, raw, 0)
        return prediction, base, item.target, valid

    def export(capture_provenance):
        capture_provenance["basePackageSha256"] = manifest["packageSha256"]
        # All previously qualified head and trunk parameters stay frozen.
        blob = real.export_weights(model)
        if blob[:5984] != weights[:5984] or blob[6144:] != weights[6144:]:
            raise ValueError("reflection training changed frozen weights")
        trained = [*manifest["trainedHeads"]]
        if "reflection" not in trained:
            trained.append("reflection")
        prior = manifest.get("provenance") or {}
        return real.write_package(args.output, model, kind="windows-capture-fit", seed=args.seed,
                                  trained_heads=trained, provenance={**prior, "reflectionFit": capture_provenance})

    return fit(args, groups, receipt, model.reflection.parameters(), forward, export)


def fit_generated(args, groups, receipt):
    model = generation.seeded(args.seed)

    def forward(item, selected=model):
        result = selected(item.inputs)
        return (result["output"], generation.analytic(item.inputs), item.target,
                result["learnable"], item.inputs.fresh[:, 3:4] > 0.5)

    def export(capture_provenance):
        return generation.write_package(args.output, model, kind="windows-capture-fit", seed=args.seed,
                                        provenance=capture_provenance)

    return fit(args, groups, receipt, model.parameters(), forward, export)


def fit(args, groups, receipt, parameters, forward, export):
    training = [g for g in groups if g.split == "train"]
    held_out = [g for g in groups if g.split == "held-out"]
    items = [x for g in training for x in g.examples]

    def loss(item):
        output, base, target, mask, *_ = forward(item)
        return objective(output, target, base, mask)

    with torch.no_grad():
        start = measure(training, forward)
        initial = sum(float(loss(item)) for item in items) / len(items)
    if all(frame["analytic"]["activeLogMae"] < 1e-7 for group in start for frame in group["frames"]):
        raise ValueError("captured training has no learnable residual against its teacher")
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    for step in range(args.steps):
        item = items[step % len(items)]
        value = loss(item)
        if not value.isfinite():
            raise ValueError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f"step {step + 1}/{args.steps}: {float(value.detach()):.8f}", flush=True)
    with torch.no_grad():
        end = measure(training, forward)
        final = sum(float(loss(item)) for item in items) / len(items)
    if not final < initial:
        raise ValueError(f"deterministic fit did not improve training loss ({initial} to {final})")
    source = provenance(receipt, groups)
    manifest, reloaded = export(source)
    # Use the held-out labels only after the weights are frozen.
    with torch.no_grad():
        reload_peak = max(float((forward(item)[0] - forward(item, reloaded)[0]).abs().max())
                          for group in groups for item in group.examples)
        independent = measure(held_out, lambda item: forward(item, reloaded))
    if reload_peak != 0:
        raise ValueError("exported model changed captured-packet CPU outputs")
    if any(frame.get("freshMaxAbs", 0) != 0 for group in end + independent for frame in group["frames"]):
        raise ValueError("generated-frame model changed trusted fresh HDR")
    report = {
        "format": "raa.neural-head-fit-report/1", "head": args.head, "seed": args.seed,
        "steps": args.steps, "learningRate": args.learning_rate, "device": "cpu",
        "torchVersion": str(torch.__version__), "weightsSha256": manifest["weightsSha256"],
        "cpuReloadMaxAbs": reload_peak,
        "provenance": source, "train": {"initialObjective": initial, "finalObjective": final,
                                       "initial": start, "final": end},
        "heldOut": {"usedForFit": False, "sequences": independent},
        "shippingQualification": {"status": "unverified", "enabledByDefault": False,
                                  "physicalLatencyAB": "not-run"},
    }
    (args.output / "fit-report.json").write_bytes(real.json_bytes(report))
    print(f"{args.head}: {initial:.8f} → {final:.8f}, {manifest['weightsSha256']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head", choices=["reflection", "frame-generation", "generation-parity"])
    parser.add_argument("capture", type=Path, help="capture manifest, or output dir for generation-parity")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--replay-warp", action="store_true",
                        help="audit and replay a legacy derived warp from original cutoff planes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    args = parser.parse_args()
    real.configure_cpu()
    torch.manual_seed(args.seed)
    if args.head == "generation-parity":
        model = generation.seeded(args.seed)
        with torch.no_grad():
            model.head.weight.uniform_(-0.15, 0.15)
            model.head.bias.uniform_(-0.15, 0.15)
        generation.write_package(args.capture, model, kind="seeded-parity-only", seed=args.seed)
        return
    if args.output is None or args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("fit requires an output path, positive steps and a positive finite learning rate")
    if args.output.exists():
        parser.error("output directory must be new")
    if args.head == "reflection":
        groups, receipt = read_capture(args.capture, "raa.reflection-teacher/1", reflection_example)
        fit_reflection(args, groups, receipt)
    else:
        corrections = {}

        def packet(root, frame):
            correction = corrections.setdefault(frame_id(frame), {}) if args.replay_warp else None
            return generated_example(root, frame, correction)

        groups, receipt = read_capture(args.capture, "raa.generated-frame-capture/1", packet)
        if args.replay_warp:
            receipt["offlineWarpReplay"] = {"kernel": "prediction_warp.wgsl", "corrections": corrections}
        fit_generated(args, groups, receipt)


if __name__ == "__main__":
    main()
