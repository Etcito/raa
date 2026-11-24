"""CPU regressions for the area-pilot trainer and its per-pixel oracle."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("pilot", Path(__file__).with_name("train-area-pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class AreaPilotTrainingTest(unittest.TestCase):
    def test_stable_reduction_retains_finite_gradient_for_subnormal_and_empty_lanes(self):
        logits = torch.zeros(1, 11, 1, 3)
        logits[:, 0] = 100.0  # Zero-weight tap dominates the spatial softmax.
        logits[:, 9] = 100.0  # Zero-weight history would round its gate to one.
        logits.requires_grad_()
        pairs = torch.zeros(1, 2, 10, 1, 3)
        pairs[:, :, 1, :, 0] = torch.tensor([0.25, 0.5]).view(1, 2, 1)
        pairs[:, :, 2, :, 0] = torch.tensor([0.0, 0.5]).view(1, 2, 1)
        pairs[:, :, 9, :, 1] = torch.tensor([0.25, 0.25]).view(1, 2, 1)
        result = pilot.stable_visibility(logits, pairs)
        torch.testing.assert_close(result, torch.tensor([[[0.25, 1.0, 1.0]]]))
        result.sum().backward()
        self.assertTrue(logits.grad.isfinite().all())
        self.assertGreater(logits.grad[0, 1, 0, 0].item(), 0.0)
        self.assertEqual(logits.grad[0, :, 0, 2].abs().sum().item(), 0.0)

    def test_stable_reduction_matches_weighted_pair_arithmetic(self):
        logits = torch.randn(1, 11, 1, 5, generator=torch.Generator().manual_seed(7))
        pairs = torch.rand(1, 2, 10, 1, 5, generator=torch.Generator().manual_seed(8))
        pairs[:, 0] *= pairs[:, 1]
        spatial = logits[:, :9].softmax(dim=1)
        history = logits[:, 9:10].sigmoid()
        current = (pairs[:, :, :9] * spatial.unsqueeze(1)).sum(dim=2)
        mixed = torch.lerp(current, pairs[:, :, 9], history)
        torch.testing.assert_close(pilot.stable_visibility(logits, pairs),
                                   mixed[:, 0] / mixed[:, 1], atol=1e-6, rtol=1e-6)

    def test_oracle_excludes_rejected_history(self):
        raw = torch.zeros(1, 12, 1, 1)
        raw[:, 1] = 1.0
        raw[:, 2:4] = 1.0
        ref = torch.full((1, 1, 1), 0.75)
        self.assertEqual(pilot.oracle(raw, ref)["oracleMse"], 0.75 ** 2)
        raw[:, 10] = 1.0
        self.assertEqual(pilot.oracle(raw, ref)["oracleMse"], 0.0)


if __name__ == "__main__":
    unittest.main()
