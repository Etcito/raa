#!/usr/bin/env python3

import dataclasses
import json
from pathlib import Path
import runpy
import struct
import tempfile
import unittest

import torch

from real_frame_model import (
    Frame, History, decode_display, golden_vectors, seeded, sha, visibility_ratio, write_package,
)

TRAIN = runpy.run_path(str(Path(__file__).with_name("train-real-frame.py")))


class RealFrameTests(unittest.TestCase):
    def test_motion_depth_normal_reactive_and_reset_gates(self):
        size = (1, 4)
        depth = torch.full((1, 1, *size), 3.0)
        normals = torch.zeros((1, 4, *size))
        normals[:, 2] = 1
        motion = torch.zeros((1, 3, *size))
        motion[:, 0, 0, 1] = 0.25
        reactive = torch.tensor([[[[1.0, 0.25, 0.0, 0.0]]]])
        validity = torch.ones_like(depth)
        frame = Frame(torch.zeros((1, 3, *size)), normals, depth, motion, reactive,
                      validity, (0.0, 0.0), (1, 7))
        old_depth = depth.clone()
        old_depth[:, :, :, 2] = 5
        old_normal = normals[:, :3].clone()
        old_normal[:, 2, :, 3] = -1
        state = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 4).expand(1, 12, 1, 4)
        previous = History(state, old_depth, old_normal, (0.25, 0.0), frame.output_size)
        model = seeded(0)
        _, warped, gate, valid = model.encode(frame, previous)
        self.assertEqual(valid.flatten().tolist(), [True, True, False, False])
        torch.testing.assert_close(gate.flatten(), torch.tensor([0, 0.75, 0, 0]))
        self.assertEqual(float(warped[0, 0, 0, 1]), 0.25)
        _, _, gate, _ = model.encode(dataclasses.replace(frame, reset=True), previous)
        self.assertEqual(float(gate.sum()), 0)
        _, _, gate, _ = model.encode(dataclasses.replace(frame, output_size=(2, 7)), previous)
        self.assertEqual(float(gate.sum()), 0)

    def test_neutral_continuous_decoder_matches_spatial_at_odd_extent(self):
        src = torch.tensor([[[[0.0, 1.0, 4.0]]]]).expand(1, 3, 1, 3)
        for output, jitter in [((3, 5), (0.15, -0.2)), ((1, 1), (-0.4, 0.1)), ((1, 3), (0, 0))]:
            frame = Frame(src, None, None, None, None, None, jitter, output)
            actual = decode_display(src, torch.zeros((1, 8, 1, 3)), output, jitter)
            torch.testing.assert_close(actual, TRAIN["spatial_baseline"](frame))
        result = visibility_ratio(torch.tensor([0.0, 0.0, 0.5]), torch.tensor([0.0, 1.0, 1.0]))
        self.assertEqual(result.tolist(), [1.0, 0.0, 0.5])

    def test_area_golden_contains_reconstructed_weighted_pairs(self):
        output = golden_vectors(seeded(0))["vectors"][1]["outputs"]
        pair = torch.tensor(output["areaPairs"]["data"]).reshape(-1, 2)
        visible = torch.tensor(output["areaVisibility"]["data"])
        torch.testing.assert_close(visible, visibility_ratio(pair[:, 0], pair[:, 1]))
        self.assertTrue(bool((pair[:, 0] <= pair[:, 1]).all()))

    def test_seeded_export_package_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seeded"
            model = seeded(0)
            manifest, reloaded = write_package(path, model, kind="seeded-parity-only", seed=0)
            self.assertEqual(manifest["trainedHeads"], [])
            self.assertEqual(len((path / "weights.fp32le").read_bytes()), 6560)
            self.assertEqual(golden_vectors(model), golden_vectors(reloaded))
            self.assertEqual(sha((path / "golden.json").read_bytes()), manifest["goldenSha256"])

    def test_display_metrics_compare_with_a_fixed_tone_map(self):
        reference = torch.rand((1, 3, 3, 5))
        perfect_psnr, perfect_ssim = TRAIN["display_metrics"](reference, reference)
        changed_psnr, changed_ssim = TRAIN["display_metrics"](0.5 * reference, reference)
        self.assertGreater(perfect_psnr, changed_psnr)
        self.assertAlmostEqual(perfect_ssim, 1.0)
        self.assertLess(changed_ssim, perfect_ssim)

    def test_joint_display_uses_composite_without_changing_late_forward_features(self):
        depth = torch.ones((1, 1, 2, 3))
        normals = torch.cat((0 * depth, 0 * depth, depth, 0 * depth), dim=1)
        hdr = depth.expand(1, 3, 2, 3)
        frame = Frame(hdr, normals, depth, torch.zeros_like(hdr), 0 * depth, depth, (0, 0), (3, 5))
        composite = dataclasses.replace(frame, display_hdr=2 * hdr)
        model = seeded(0)
        before, _ = model(frame)
        after, _ = model(composite)
        torch.testing.assert_close(before["state"], after["state"])
        torch.testing.assert_close(before["reflection"], after["reflection"])
        self.assertTrue(bool((after["display"] > before["display"]).all()))
        torch.testing.assert_close(TRAIN["spatial_baseline"](composite),
                                   2 * TRAIN["spatial_baseline"](frame))

    def test_joint_display_loss_has_finite_gradient_at_black(self):
        hdr = torch.zeros((1, 3, 2, 3), requires_grad=True)
        TRAIN["fixed_display"](hdr).sum().backward()
        self.assertTrue(bool(hdr.grad.isfinite().all()))
        self.assertTrue(bool((hdr.grad > 0).all()))

    def test_final_split_rejects_renamed_replays_and_duplicate_planes(self):
        split = {"groupIds": ["dev"], "replayHashes": ["same tape"],
                 "frameIds": ["dev-0"], "fingerprints": [("a", "b", "c")]}
        heldout = {"groupIds": ["final"], "replayHashes": ["same tape"],
                   "frameIds": ["final-0"], "fingerprints": [("z", "y", "x")]}
        with self.assertRaisesRegex(ValueError, "replayHashes"):
            TRAIN["require_separate"](split, heldout)
        heldout["replayHashes"] = ["new tape"]
        TRAIN["require_separate"](split, heldout)
        heldout["fingerprints"] = [["a", "b", "c"]]
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            TRAIN["require_separate"](split, heldout)

    def test_reflection_change_excludes_invalid_transition_and_reactive_resets(self):
        shape = (1, 4)
        motion = torch.zeros((1, 3, *shape))
        reactive = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
        reflect = torch.zeros((1, 6, *shape))
        reflect[:, 3:5] = 1
        previous = reflect[:, 3:5].clone()
        previous[:, 0, 0, 1] = 0
        previous[:, 1, 0, 3] = 2
        valid = torch.ones((1, 1, *shape), dtype=torch.bool)
        residual = torch.ones((1, 3, *shape))
        frame = Frame(residual, None, None, motion, reactive, None, (0, 0), shape,
                      reflection=reflect)
        old = (torch.zeros_like(residual), previous, (0.0, 0.0))
        delta, keep = TRAIN["reflection_residual_change"](frame, residual, old, valid)
        torch.testing.assert_close(delta, residual)
        self.assertEqual(keep.flatten().tolist(), [True, False, False, False])
        self.assertIsNone(TRAIN["reflection_residual_change"](
            dataclasses.replace(frame, reset=True), residual, old, valid))

    def test_native_replay_uses_exact_clock_jitter_and_frozen_dispatch(self):
        hdr = torch.zeros((1, 3, 3, 4))
        frame = Frame(hdr, None, None, None, None, None, (0.25, -0.375), (6, 8))
        sample = TRAIN["Sample"]("final-01", frame, None, None, None, None, (),
                                 event="warm-step", time_ms=1020)
        record = {"id": sample.id, "arm": "joint", "frameId": 17, "timeMs": sample.time_ms,
                  "event": sample.event, "jitterNdc": [0.125, 0.25],
                  "dispatch": {"weightsSha256": "frozen", "realFrame": 17,
                               "ran": {"area": True, "reflection": True, "sr": True}}}
        verify = TRAIN["verify_replay_frame"]
        verify(record, sample, "joint", "frozen")
        sr_reflection = {**record, "arm": "sr-reflection", "dispatch": {
            **record["dispatch"], "ran": {**record["dispatch"]["ran"], "area": False}}}
        verify(sr_reflection, sample, "sr-reflection", "frozen")
        with self.assertRaisesRegex(ValueError, "requested heads"):
            verify({**record, "arm": "sr-reflection"}, sample, "sr-reflection", "frozen")
        with self.assertRaisesRegex(ValueError, "teacher clock"):
            verify({**record, "timeMs": 1021}, sample, "joint", "frozen")
        with self.assertRaisesRegex(ValueError, "jitter"):
            verify({**record, "jitterNdc": [0, 0]}, sample, "joint", "frozen")
        with self.assertRaisesRegex(ValueError, "frozen checkpoint"):
            verify(record, sample, "joint", "other")

    def test_loader_rejects_torn_planes_and_non_windows_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            data = struct.pack("<3f", 1, 2, 3)
            (path / "plane.bin").write_bytes(data)
            descriptor = {"file": "plane.bin", "shape": [1, 1, 3], "dtype": "f32le",
                          "byteLength": len(data), "sha256": sha(data)}
            self.assertTrue(torch.equal(TRAIN["read_plane"](path, descriptor).flatten(),
                                        torch.tensor([1.0, 2.0, 3.0])))
            with self.assertRaisesRegex(ValueError, "damaged plane"):
                TRAIN["read_plane"](path, {**descriptor, "sha256": "bad"})
            capture = path / "manifest.json"
            capture.write_text(json.dumps({"format": TRAIN["CAPTURE_FORMAT"], "source": {}}))
            with self.assertRaisesRegex(ValueError, "Windows capture"):
                TRAIN["load_capture"](capture)


if __name__ == "__main__":
    unittest.main()
