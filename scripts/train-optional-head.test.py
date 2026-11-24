"""CPU regressions for the optional head capture, fit and FP32 export."""

import importlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from optional_head_capture import CHANNELS, json_bytes, read_capture, sha, write_plane

diagnostic = importlib.import_module("generate-optional-head-diagnostic")
trainer = importlib.import_module("train-optional-head")


class OptionalHeadTrainingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.capture = self.root / "diagnostic" / "manifest.json"
        diagnostic.generate(self.capture.parent)

    def edit_manifest(self):
        return json.loads(self.capture.read_bytes())

    def test_independent_overfits_and_export_replay(self):
        capture_hash = sha(self.capture.read_bytes())
        seeds = set()
        for signal, (input_channels, output_channels) in CHANNELS.items():
            with self.subTest(signal=signal):
                fit = self.root / signal
                trainer.train(self.capture, fit, signal, 300, 0.01)
                report = json.loads((fit / "training-report.json").read_bytes())
                self.assertEqual(report["captureSha256"], capture_hash)
                self.assertEqual(report["captureSource"]["kind"], "cpu-diagnostic")
                self.assertEqual(report["signal"], signal)
                self.assertEqual(report["device"], "cpu")
                self.assertLess(report["finalLoss"], 0.01)
                self.assertLess(report["finalLoss"], report["initialLoss"] * 0.1)
                self.assertLess(report["finalLoss"], report["inputLoss"] * 0.1)
                seeds.add(report["initializationSeed"])
                for frame in report["frames"]:
                    self.assertLess(frame["mse"], frame["inputMse"])
                    self.assertGreater(frame["slices"]["missingSource"]["pixels"], 0)
                    self.assertGreater(frame["slices"]["freshlyTraced"]["pixels"], 0)
                    self.assertGreater(frame["slices"]["validReused"]["pixels"], 0)
                    self.assertEqual(len(frame["referenceMetadata"]["seedBatches"]), 2)
                self.assert_replay(fit, input_channels, output_channels)
        self.assertEqual(len(seeds), len(CHANNELS))

    def assert_replay(self, fit, input_channels, output_channels):
        manifest = json.loads((fit / "manifest.json").read_bytes())
        descriptor = manifest["weights"]
        blob = (fit / descriptor["file"]).read_bytes()
        self.assertEqual(sha(blob), descriptor["sha256"])
        self.assertEqual(len(blob), descriptor["byteLength"])
        weights = {}
        for tensor in descriptor["tensors"]:
            count = 1
            for extent in tensor["shape"]:
                count *= extent
            data = blob[tensor["offset"]:tensor["offset"] + 4 * count]
            self.assertEqual(sha(data), tensor["sha256"])
            weights[tensor["name"]] = torch.tensor(struct.unpack(f"<{count}f", data)).reshape(tensor["shape"])
        self.assertEqual(manifest["inputChannels"], list(input_channels))
        self.assertEqual(manifest["outputChannels"], list(output_channels))
        vectors = json.loads((fit / "trainer-vectors.json").read_bytes())
        for vector in vectors:
            raw = torch.tensor(vector["input"]).reshape(-1, len(input_channels))
            mask = raw[:, input_channels.index("metricDepth")] > 0
            output = torch.full((len(raw), len(output_channels)), manifest["backgroundValue"])
            x = raw[mask] * torch.tensor(manifest["normalizationScales"])
            x = F.linear(x, weights["hidden.weight"], weights["hidden.bias"]).relu()
            logits = F.linear(x, weights["output.weight"], weights["output.bias"])
            if manifest["outputActivation"] == "softplus":
                output[mask] = F.softplus(logits)
                self.assertGreater(output.max().item(), 1.0)
            else:
                output[mask] = logits.sigmoid()
                self.assertTrue(((output >= 0) & (output <= 1)).all())
            torch.testing.assert_close(output.flatten(), torch.tensor(vector["output"]), rtol=0, atol=0)

    def test_determinism_and_signal_subset(self):
        repeat = self.root / "repeat"
        diagnostic.generate(repeat)
        for path in self.capture.parent.iterdir():
            self.assertEqual(path.read_bytes(), (repeat / path.name).read_bytes())
        manifest = self.edit_manifest()
        signal = "ao-visibility"
        del manifest["frames"][1]["signals"][signal]
        self.capture.write_bytes(json_bytes(manifest))
        outputs = [self.root / "fit-a", self.root / "fit-b"]
        for fit in outputs:
            trainer.train(self.capture, fit, signal, 25, 0.01)
        for path in outputs[0].iterdir():
            self.assertEqual(path.read_bytes(), (outputs[1] / path.name).read_bytes())
        report = json.loads((outputs[0] / "training-report.json").read_bytes())
        self.assertEqual([frame["id"] for frame in report["frames"]], ["diagnostic-0"])

    def test_seed_overlap_and_unconverged_reference_are_rejected(self):
        signal = "ao-visibility"
        manifest = self.edit_manifest()
        signals = manifest["frames"][0]["signals"]
        record = signals[signal]
        reference = record["referenceMetadata"]
        batches = reference["seedBatches"]
        other_seed = signals["indirect-diffuse"]["inputSeeds"][0]
        original = batches[1][0]
        for colliding_seed in (record["inputSeeds"][0], batches[0][0], other_seed):
            batches[1][0] = colliding_seed
            self.capture.write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                read_capture(self.capture, signal)
        batches[1][0] = original
        reference["batchRmseTolerance"] = 0
        self.capture.write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "exceeds its tolerance"):
            read_capture(self.capture, signal)

    def test_plane_digest_and_mask_are_checked(self):
        signal = "indirect-specular"
        manifest = self.edit_manifest()
        record = manifest["frames"][0]["signals"][signal]
        path = self.capture.parent / record["input"]["file"]
        original = path.read_bytes()
        path.write_bytes(b"\xff" + original[1:])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            read_capture(self.capture, signal)
        values = list(struct.unpack(f"<{len(original) // 4}f", original))
        values[CHANNELS[signal][0].index("freshlyTraced")] = 1.0
        record["input"] = write_plane(path.parent, path.name, record["input"]["shape"], values)
        self.capture.write_bytes(json_bytes(manifest))
        output = self.root / "invalid"
        with self.assertRaisesRegex(ValueError, "inconsistent source validity masks"):
            trainer.train(self.capture, output, signal, 1, 0.01)
        self.assertFalse(output.exists())

    def test_relabelling_diagnostic_requires_windows_provenance(self):
        manifest = self.edit_manifest()
        manifest["source"].update(kind="windows-render", buildId="test-only",
                                  facts=dict(hardware=False, browser="CPU diagnostic",
                                             adapter=dict(isFallbackAdapter=True)))
        self.capture.write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "Windows render provenance"):
            read_capture(self.capture, "ao-visibility")


if __name__ == "__main__":
    unittest.main()
