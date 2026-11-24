#!/usr/bin/env python3

import copy
from pathlib import Path
import runpy
import struct
import tempfile
import unittest

import torch

import generated_frame_model as generated

TRAIN = runpy.run_path(str(Path(__file__).with_name("train-neural-heads.py")))


class HeadTests(unittest.TestCase):
    def capture(self):
        return {
            "sourceAvailableMs": 3, "guidesAvailableMs": 5, "cutoffMs": 6,
            "targetRenderStartedMs": 8, "sourceRealFrameId": 2, "targetRealFrameId": 3,
            "sourceAgeRealFrames": 1,
            "sourceSnapshot": {"revision": 1, "target": {"tick": 4}},
            "acceptedTarget": {"revision": 1, "target": {"tick": 5}, "observedAtMs": 6,
                               "inputs": {"cutoffMs": 6, "inputIds": ["a", None],
                                          "confirmedMask": [True, False], "receivedAtMs": [4, None]}},
        }

    def test_late_receipt_teacher_and_revision_guards(self):
        packet = self.capture()
        self.assertEqual(TRAIN["audit_cutoff"](packet), 1)
        late = copy.deepcopy(packet)
        late["acceptedTarget"]["inputs"]["receivedAtMs"][0] = 7
        with self.assertRaisesRegex(ValueError, "post-cutoff input"):
            TRAIN["audit_cutoff"](late)
        with self.assertRaisesRegex(ValueError, "after the cutoff"):
            TRAIN["audit_cutoff"]({**packet, "targetRenderStartedMs": 4})
        crossed = copy.deepcopy(packet)
        crossed["acceptedTarget"]["revision"] = 2
        with self.assertRaisesRegex(ValueError, "revision"):
            TRAIN["audit_cutoff"](crossed)
        self.assertEqual(packet["acceptedTarget"]["inputs"]["receivedAtMs"][0], 4)

    def test_warp_audit_rejects_a_reuse_for_changed_pixel(self):
        source_hdr = torch.ones((1, 4, 1, 3))
        source_depth = torch.full((1, 1, 1, 3), 0.3)
        motion = torch.cat((torch.zeros((1, 3, 1, 3)), source_depth), dim=1)
        lighting = torch.tensor([[[[0.0, 1.0, 0.0]]]])
        warp = source_hdr.clone()
        warp[:, :, 0, 1] = 0
        fresh = source_hdr - warp
        packet = generated.Packet(warp, fresh, source_depth, motion, lighting, 1)
        TRAIN["audit_warp"](packet, source_hdr, source_depth)
        packet.warp = source_hdr
        with self.assertRaisesRegex(ValueError, "admitted runtime guides"):
            TRAIN["audit_warp"](packet, source_hdr, source_depth)
        correction = {}
        TRAIN["audit_warp"](packet, source_hdr, source_depth, correction)
        self.assertEqual(correction, {"warpMismatchPixels": 1, "excludedFreshPixels": 0})
        self.assertEqual(float(packet.warp[0, 3, 0, 1]), 0)
        self.assertEqual(float(packet.fresh[0, 3, 0, 1]), 1)

    def test_model_learns_only_on_accepted_stale_warp(self):
        model = generated.seeded(0)
        with torch.no_grad():
            model.head.bias.fill_(0.5)
        shape = (1, 1, 1, 3)
        warp = torch.ones((1, 4, 1, 3))
        warp[:, :, 0, :2] = 0
        fresh = torch.zeros_like(warp)
        fresh[:, :, 0, 0] = torch.tensor([0.2, 0.4, 0.6, 1])
        inputs = generated.Packet(warp, fresh, torch.full(shape, 0.25),
                                  torch.zeros((1, 4, 1, 3)), torch.zeros(shape), 1)
        result = model(inputs)
        torch.testing.assert_close(result["output"][:, :, 0, 0], fresh[:, :3, 0, 0])
        self.assertFalse(bool(result["coverage"][0, 0, 0, 1]))
        self.assertEqual(float(result["output"][:, :, 0, 1].detach().sum()), 0)
        self.assertTrue(bool(result["learnable"][0, 0, 0, 2]))
        self.assertTrue(bool((result["output"][:, :, 0, 2] > 1).all()))

    def test_reflection_surface_and_independent_teacher_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planes = {}
            for name, values in {
                "hdr": [0.2, 0.3, 0.4],
                "guide": [0, 0, 1, 0.08, 0.5, 0, 0, 0, 0, 0, 0, 1],
                "reflection": [0.2, 0.3, 0.4, 1, 1, 0],
                "reflectionTarget": [0.21, 0.31, 0.41],
                "depth": [1],
                "motion": [0, 0, 0],
            }.items():
                blob = struct.pack(f"<{len(values)}f", *values)
                file = name + ".bin"
                (root / file).write_bytes(blob)
                planes[name] = {"file": file, "shape": [1, 1, len(values)], "dtype": "f32le",
                                "byteLength": len(blob), "sha256": generated.sha(blob)}
            frame = {"id": "fixture", "planes": planes, "inputSeeds": [],
                     "teacher": {"seeds": [[23], [37]],
                                 "reflection": {"logRmse": 0.01, "tolerance": 0.03, "accepted": 1}}}
            actual = TRAIN["reflection_example"](root, frame, native_screen=True)
            self.assertEqual(actual.id, "fixture")
            frame["teacher"]["seeds"][1] = [23]
            with self.assertRaisesRegex(ValueError, "overlapping"):
                TRAIN["reflection_example"](root, frame, native_screen=True)
            frame["teacher"]["seeds"][1] = [37]
            frame["teacher"]["reflection"]["accepted"] = 0
            with self.assertRaisesRegex(ValueError, "screen support"):
                TRAIN["reflection_example"](root, frame, native_screen=True)

    def test_seeded_package_preserves_model_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            model = generated.seeded(3)
            path = Path(directory) / "seeded"
            manifest, loaded = generated.write_package(path, model, kind="seeded-parity-only", seed=3)
            self.assertEqual(manifest["trainedHeads"], [])
            self.assertEqual(manifest["weightsByteLength"], 1984)
            self.assertEqual(generated.golden_vectors(model), generated.golden_vectors(loaded))


if __name__ == "__main__":
    unittest.main()
