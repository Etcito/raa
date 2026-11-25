"""Device-free checks for the spherical diffuse gate package and capture contract."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location("grid_training", Path(__file__).with_name("train-grid-diffuse.py"))
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def example_taps(count=3):
    features = np.zeros((count, 8, 8), np.float32)
    weights = np.zeros((count, 8), np.float32)
    directional = np.zeros((count, 8, 3), np.float32)
    features[:, :2, :] = [[0.2, 0, 0, 1, 0.1, 0.5, 0.8, 1], [-0.2, 0, 0, 1, 0.1, 0.5, 0.8, 1]]
    weights[:, :2] = 0.4
    directional[:, :2, :] = [[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]]
    legacy = np.full((count, 3), 0.5, np.float32)
    teacher = np.full((count, 3), 0.8, np.float32)
    return grid.Taps(features, weights, directional, legacy, teacher, teacher.copy(),
                     np.ones(count, np.float32), np.ones(count, np.float32))


def sample_capture(directory, scene, split):
    taps = example_taps()
    rows = np.zeros((len(taps.features), 128), dtype="<f4")
    rows.view("<u4")[:, 11] = 2
    rows[:, 12:15] = taps.teacher_a
    rows[:, 16:19] = taps.teacher_b
    rows[:, 15] = rows[:, 19] = 1
    rows[:, 20:23] = taps.legacy
    rows[:, 28:31] = grid.gated_mix(taps)
    rows[:, 32:] = np.concatenate(
        (taps.features, taps.directional, taps.weights[..., None]), axis=-1
    ).reshape(-1, 96)
    data = rows.tobytes()
    directory.mkdir()
    (directory / "frame.bin").write_bytes(data)
    path = directory / "capture.json"
    grid.write_json(path, {
        "format": grid.CAPTURE_FORMAT, "units": grid.UNITS,
        "scene": {"id": scene, "split": split, "geometrySha256": grid.sha256(scene.encode())},
        "source": {"platform": "win32", "adapter": "test fixture"},
        "frames": [{
            "id": "one", "condition": "static", "file": "frame.bin", "rows": len(rows),
            "sha256": grid.sha256(data), "reference": {
                "samplesPerBank": 32, "independent": True,
                "firstHitSha256": grid.sha256(rows.view("<u4")[:, [0, 1, 2, 3, 31]].tobytes()),
            },
        }],
    })
    return path


class GridDiffuseTrainerTest(unittest.TestCase):
    def test_weight_package_uses_renderer_offsets(self):
        data = (np.arange(81, dtype="<f4") / 8).tobytes()
        w1, b1, w2, b2 = grid.unpack(data)
        self.assertEqual(w1[1, 0], 1)
        self.assertEqual(b1[0], 8)
        self.assertEqual(w2[0], 9)
        self.assertEqual(b2[0], 10)
        self.assertEqual(grid.pack((w1, b1, w2, b2)), data)
        with self.assertRaisesRegex(ValueError, "bounded by 16"):
            grid.unpack(np.full(81, 17, dtype="<f4").tobytes())

    def test_one_cell_reweighting_cancels_and_routes_to_fallback(self):
        taps = example_taps()
        taps.weights[0, 1] = 0
        taps.weights[1] = 0
        taps.blend[:] = [0, 0, 0.5]
        params = tuple(np.ones(shape, np.float32) for _, shape in grid.LAYOUT)
        predicted = grid.gated_mix(taps, params)
        np.testing.assert_array_equal(predicted[:2], 0)
        np.testing.assert_array_equal(
            grid.gated_mix(taps, params, require_support=False)[0], taps.directional[0, 0],
        )
        report = grid.coverage(taps, params)
        self.assertEqual(report["neighborCountHistogram0To8"][:3], [1, 1, 1])
        self.assertEqual(report["fallbackFull"], 2)
        self.assertEqual(report["fallbackPartial"], 1)
        self.assertEqual(report["oneNeighborNormalizedRgbDeltaMax"], 0)

    def test_empty_frame_does_not_drop_coverage_or_invent_a_score(self):
        taps, rows = grid.unpack_rows(b"", 0)
        params = tuple(np.zeros(shape, np.float32) for _, shape in grid.LAYOUT)
        self.assertEqual(rows.shape, (0, 128))
        self.assertEqual(grid.score(taps, params, 1), {"hits": 0, "supportedHits": 0, "coverage": None})
        self.assertEqual(grid.coverage(taps, params)["neighborCountHistogram0To8"], [0] * 9)

    def test_analytic_gradient_matches_finite_difference(self):
        taps = example_taps()
        rng = np.random.default_rng(5)
        params = [rng.normal(0, 0.3, shape).astype(np.float32) for _, shape in grid.LAYOUT]
        base_loss, derivatives = grid.loss_gradient(taps, params, 0.5)
        self.assertAlmostEqual(base_loss, grid.log_error(grid.gated_mix(taps, params), taps.teacher, 0.5))
        index = (1, 0)
        step = 0.005
        params[0][index] += step
        up = grid.log_error(grid.gated_mix(taps, params), taps.teacher, 0.5)
        params[0][index] -= 2 * step
        down = grid.log_error(grid.gated_mix(taps, params), taps.teacher, 0.5)
        self.assertAlmostEqual(float(derivatives[0][index]), (up - down) / (2 * step), places=5)

    def test_freezes_checkpoint_before_scoring_another_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            training = sample_capture(directory / "train", "cornell", "train")
            held = sample_capture(directory / "held", "atrium", "test")
            checkpoint = directory / "checkpoint"
            grid.fit(SimpleNamespace(capture=training, output=checkpoint, steps=150, batch=3,
                                     seed=9, learning_rate=0.01))
            freeze = json.loads((checkpoint / "freeze.json").read_bytes())
            digest = freeze["weightsSha256"]
            report_path = directory / "score.json"
            grid.evaluate(SimpleNamespace(model=checkpoint, capture=[held], output=report_path,
                                          weights_sha256=digest))
            report = json.loads(report_path.read_bytes())
            self.assertEqual(report["weightsSha256"], digest)
            quality = report["reserved"][0]["all"]
            self.assertLess(quality["directionalLearned"]["rgbRmse"], quality["directionalAnalytic"]["rgbRmse"])
            with self.assertRaisesRegex(ValueError, "pre-registered"):
                grid.evaluate(SimpleNamespace(model=checkpoint, capture=[directory / "missing"],
                                              output=report_path, weights_sha256="0" * 64))
            (held.parent / "frame.bin").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                grid.read_capture(held, "test")


if __name__ == "__main__":
    unittest.main()
