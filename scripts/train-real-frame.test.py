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

    def test_seeded_export_package_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seeded"
            model = seeded(0)
            manifest, reloaded = write_package(path, model, kind="seeded-parity-only", seed=0)
            self.assertEqual(manifest["trainedHeads"], [])
            self.assertEqual(len((path / "weights.fp32le").read_bytes()), 6560)
            self.assertEqual(golden_vectors(model), golden_vectors(reloaded))
            self.assertEqual(sha((path / "golden.json").read_bytes()), manifest["goldenSha256"])

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
