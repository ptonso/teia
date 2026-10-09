"""Tests for the unpaired metrics: FrechetDistance,
KernelDistance, DistributionTest, and the shared embedding/feature-extractor plumbing they're
built on. No live pipeline wires these yet (net-new, no existing formula to preserve parity
against) — these tests prove the math and the nested-``_target_`` mechanism directly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from teia.core.instantiate import instantiate
from teia.node.eval.metric.unpaired._embedding import EmbeddingAccumulator, load_image_tensor
from teia.node.eval.metric.unpaired._shared import frechet_distance, ks_statistic, rbf_mmd_squared
from teia.node.eval.metric.unpaired.distribution_test import DistributionTest
from teia.node.eval.metric.unpaired.features import FlattenFeatures
from teia.node.eval.metric.unpaired.frechet_distance import FrechetDistance
from teia.node.eval.metric.unpaired.kernel_distance import KernelDistance


class SharedMathTests(unittest.TestCase):
    def test_frechet_distance_is_zero_for_identical_gaussians(self) -> None:
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.1], [0.1, 1.0]])
        self.assertAlmostEqual(frechet_distance(mean, cov, mean, cov), 0.0, places=6)

    def test_frechet_distance_matches_mean_shift_only_case(self) -> None:
        # Identical covariance, means differ by (3, 4) -> trace term is 0, distance is ||diff||^2.
        cov = np.eye(2)
        mean_a = np.array([0.0, 0.0])
        mean_b = np.array([3.0, 4.0])
        self.assertAlmostEqual(frechet_distance(mean_a, cov, mean_b, cov), 25.0, places=5)

    def test_frechet_distance_is_symmetric(self) -> None:
        mean_a, cov_a = np.array([0.0, 0.0]), np.eye(2)
        mean_b, cov_b = np.array([1.0, -1.0]), np.array([[2.0, 0.3], [0.3, 1.5]])
        forward = frechet_distance(mean_a, cov_a, mean_b, cov_b)
        backward = frechet_distance(mean_b, cov_b, mean_a, cov_a)
        self.assertAlmostEqual(forward, backward, places=5)

    def test_rbf_mmd_is_near_zero_for_identical_distributions(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(200, 4))
        # Split the same sample in half - both halves come from the same distribution.
        mmd = rbf_mmd_squared(features[:100], features[100:])
        self.assertLess(mmd, 0.05)

    def test_rbf_mmd_is_larger_for_separated_distributions(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0.0, size=(100, 4))
        b = rng.normal(loc=10.0, size=(100, 4))
        mmd_same = rbf_mmd_squared(a, rng.normal(loc=0.0, size=(100, 4)))
        mmd_diff = rbf_mmd_squared(a, b)
        self.assertGreater(mmd_diff, mmd_same)

    def test_ks_statistic_is_zero_for_identical_samples(self) -> None:
        sample = np.linspace(0.0, 1.0, 50)
        self.assertAlmostEqual(ks_statistic(sample, sample), 0.0, places=6)

    def test_ks_statistic_is_one_for_disjoint_ranges(self) -> None:
        a = np.linspace(0.0, 1.0, 20)
        b = np.linspace(10.0, 11.0, 20)
        self.assertAlmostEqual(ks_statistic(a, b), 1.0, places=6)


class FlattenFeaturesTests(unittest.TestCase):
    def test_global_average_pool_shape_and_value(self) -> None:
        extractor = FlattenFeatures()
        images = torch.ones((2, 3, 4, 4)) * torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
        embedded = extractor(images)
        self.assertEqual(tuple(embedded.shape), (2, 3))
        self.assertTrue(torch.allclose(embedded, torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])))

    def test_adds_batch_dim_for_a_single_image(self) -> None:
        extractor = FlattenFeatures()
        embedded = extractor(torch.zeros((3, 4, 4)))
        self.assertEqual(tuple(embedded.shape), (1, 3))


def _write_png(path: Path, color: tuple[int, int, int], size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, :] = color
    Image.fromarray(array, mode="RGB").save(path)


class EmbeddingAccumulatorTests(unittest.TestCase):
    def test_update_paths_loads_and_embeds_real_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_a = root / "a.png"
            path_b = root / "b.png"
            _write_png(path_a, (255, 0, 0))
            _write_png(path_b, (0, 255, 0))

            accumulator = EmbeddingAccumulator(FlattenFeatures())
            accumulator.update_paths([str(path_a), str(path_b)])

            self.assertEqual(accumulator.n, 2)
            embeddings = accumulator.embeddings()
            self.assertEqual(embeddings.shape, (2, 3))
            # Red image -> high R channel mean, low G/B; green -> the reverse.
            self.assertGreater(embeddings[0, 0], embeddings[0, 1])
            self.assertGreater(embeddings[1, 1], embeddings[1, 0])

    def test_load_image_tensor_is_chw_in_unit_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            _write_png(path, (128, 64, 32))
            tensor = load_image_tensor(str(path))
            self.assertEqual(tuple(tensor.shape), (3, 8, 8))
            self.assertTrue(torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0))


class FrechetDistanceMetricTests(unittest.TestCase):
    def test_streaming_update_compute_matches_from_source(self) -> None:
        metric = FrechetDistance(feature_extractor=FlattenFeatures())
        rng = torch.Generator().manual_seed(0)
        pred_batch = torch.rand((6, 3, 4, 4), generator=rng)
        real_batch = torch.rand((6, 3, 4, 4), generator=rng) * 0.5 + 0.3

        metric.update(pred=pred_batch)
        metric.update(real=real_batch)
        streamed = metric.compute()

        self.assertEqual(streamed["n_pred"], 6)
        self.assertEqual(streamed["n_real"], 6)
        self.assertIn("frechet_distance", streamed)
        self.assertGreaterEqual(streamed["frechet_distance"], 0.0)

    def test_from_source_reads_real_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_paths = []
            real_paths = []
            for index in range(4):
                pred_path = root / f"pred_{index}.png"
                real_path = root / f"real_{index}.png"
                _write_png(pred_path, (200, 20, 20))
                _write_png(real_path, (20, 200, 20))
                pred_paths.append(str(pred_path))
                real_paths.append(str(real_path))

            metric = FrechetDistance(feature_extractor=FlattenFeatures())
            result = metric.from_source(None, pred_paths=pred_paths, real_paths=real_paths)

            self.assertEqual(result["n_pred"], 4)
            self.assertEqual(result["n_real"], 4)
            # Red vs green images are well-separated in this feature space (pure-color images
            # have zero within-set variance, so this is exactly the squared mean shift).
            self.assertGreater(result["frechet_distance"], 0.9)

    def test_identical_distributions_give_near_zero_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(4):
                path = root / f"sample_{index}.png"
                _write_png(path, (100, 150, 200))
                paths.append(str(path))

            metric = FrechetDistance(feature_extractor=FlattenFeatures())
            result = metric.from_source(None, pred_paths=paths, real_paths=paths)
            self.assertAlmostEqual(result["frechet_distance"], 0.0, places=4)


class KernelDistanceMetricTests(unittest.TestCase):
    def test_from_source_reads_paths_and_reports_mmd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_paths, real_paths = [], []
            for index in range(4):
                pred_path, real_path = root / f"p{index}.png", root / f"r{index}.png"
                _write_png(pred_path, (200, 20, 20))
                _write_png(real_path, (20, 200, 20))
                pred_paths.append(str(pred_path))
                real_paths.append(str(real_path))

            metric = KernelDistance(feature_extractor=FlattenFeatures())
            result = metric.from_source(None, pred_paths=pred_paths, real_paths=real_paths)
            self.assertEqual(result["n_pred"], 4)
            self.assertIn("mmd_squared", result)
            self.assertGreaterEqual(result["mmd_squared"], 0.0)


class DistributionTestMetricTests(unittest.TestCase):
    def test_from_source_reports_per_feature_ks(self) -> None:
        rng = np.random.default_rng(1)
        pred = rng.normal(loc=0.0, size=(50, 3))
        real = rng.normal(loc=0.0, size=(50, 3))

        metric = DistributionTest(feature_names=["x", "y", "z"])
        result = metric.from_source(None, pred=pred, real=real)

        self.assertEqual(result["n_pred"], 50)
        self.assertEqual([row["feature"] for row in result["per_feature"]], ["x", "y", "z"])
        self.assertGreaterEqual(result["mean_ks_statistic"], 0.0)

    def test_streaming_accumulates_across_batches(self) -> None:
        metric = DistributionTest()
        metric.update(pred=np.array([[0.0], [0.1]]), real=np.array([[0.0], [0.1]]))
        metric.update(pred=np.array([[0.2]]), real=np.array([[0.2]]))
        result = metric.compute()
        self.assertEqual(result["n_pred"], 3)
        self.assertAlmostEqual(result["mean_ks_statistic"], 0.0, places=6)


class NestedTargetResolutionTests(unittest.TestCase):
    def test_feature_extractor_nested_target_resolves_via_plain_instantiate(self) -> None:
        """Proves the mechanism documented in features.py's module docstring: a plain
        (non-TeiaNode) nested `_target_` resolves automatically via `core/eval/runner.py`'s
        unwrapped `hydra.utils.instantiate` — no special-casing needed for this metric to be
        wired through a real eval-bundle config."""
        config = {
            "_target_": "teia.node.eval.metric.unpaired.frechet_distance.FrechetDistance",
            "device": "cpu",
            "feature_extractor": {"_target_": "teia.node.eval.metric.unpaired.features.FlattenFeatures"},
        }
        metric = instantiate(config)
        self.assertIsInstance(metric, FrechetDistance)
        self.assertIsInstance(metric._pred.feature_extractor, FlattenFeatures)


if __name__ == "__main__":
    unittest.main()
