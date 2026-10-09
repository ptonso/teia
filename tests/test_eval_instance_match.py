"""Tests for the instance-match collapse: the 4 similarity kernels and
``InstanceMatch``, wired end-to-end into ``RankedAveragePrecision``'s ``matched=`` path."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from teia.core.eval.runner import RunEvalSource
from teia.core.capture.store import CaptureWriter
from teia.node.eval.kernel.iou_mask import IouMask
from teia.node.eval.kernel.iou_rotated import IouRotated
from teia.node.eval.kernel.iou_xyxy import IouXyxy
from teia.node.eval.kernel.oks import Oks
from teia.node.eval.metric.aligned.instance_match import InstanceMatch
from teia.node.eval.metric.aligned.ranked_average_precision import RankedAveragePrecision


_DET_KEYS = {
    "pred": {"geometry": "capture.det.boxes", "category": "capture.det.category", "score": "capture.det.score", "index": "capture.det.sample_idx"},
    "gt": {"geometry": "batch.bboxes", "category": "batch.cls", "index": "batch.sample_idx"},
}


class KernelTests(unittest.TestCase):
    def test_iou_xyxy_identical_boxes_is_one(self) -> None:
        kernel = IouXyxy()
        self.assertAlmostEqual(kernel((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)), 1.0, places=6)

    def test_iou_xyxy_disjoint_boxes_is_zero(self) -> None:
        kernel = IouXyxy()
        self.assertEqual(kernel((0.1, 0.1, 0.05, 0.05), (0.9, 0.9, 0.05, 0.05)), 0.0)

    def test_iou_xyxy_partial_overlap_matches_hand_computed_value(self) -> None:
        kernel = IouXyxy()
        # Two unit squares offset by 0.5 on each axis: intersection 0.5x0.5=0.25, union 1.75.
        got = kernel((0.5, 0.5, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(got, 0.25 / 1.75, places=6)

    def test_iou_rotated_identical_boxes_is_one(self) -> None:
        kernel = IouRotated()
        self.assertAlmostEqual(kernel((0.5, 0.5, 0.3, 0.2, 0.4), (0.5, 0.5, 0.3, 0.2, 0.4)), 1.0, places=5)

    def test_iou_rotated_reduces_to_axis_aligned_iou_at_zero_angle(self) -> None:
        rotated = IouRotated()((0.5, 0.5, 0.4, 0.2, 0.0), (0.6, 0.5, 0.4, 0.2, 0.0))
        axis_aligned = IouXyxy()((0.5, 0.5, 0.4, 0.2), (0.6, 0.5, 0.4, 0.2))
        self.assertAlmostEqual(rotated, axis_aligned, places=5)

    def test_iou_mask_identical_polygons_is_one(self) -> None:
        square = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        self.assertAlmostEqual(IouMask()(square, square), 1.0, places=6)

    def test_iou_mask_disjoint_polygons_is_zero(self) -> None:
        a = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
        b = [(0.9, 0.9), (1.0, 0.9), (1.0, 1.0), (0.9, 1.0)]
        self.assertEqual(IouMask()(a, b), 0.0)

    def test_oks_identical_keypoints_is_near_one(self) -> None:
        kpts = [(0.3, 0.3, 2.0), (0.5, 0.5, 2.0), (0.7, 0.3, 2.0)]
        self.assertAlmostEqual(Oks()(kpts, kpts), 1.0, places=6)

    def test_oks_ignores_invisible_ground_truth_points(self) -> None:
        gt = [(0.3, 0.3, 0.0), (0.5, 0.5, 2.0)]  # first point not visible
        pred_matches_only_visible = [(0.0, 0.0, 0.0), (0.5, 0.5, 2.0)]
        self.assertAlmostEqual(Oks()(pred_matches_only_visible, gt), 1.0, places=6)

    def test_oks_decreases_with_distance(self) -> None:
        gt = [(0.5, 0.5, 2.0), (0.6, 0.5, 2.0)]
        close = Oks()([(0.51, 0.5, 2.0), (0.6, 0.5, 2.0)], gt)
        far = Oks()([(0.9, 0.9, 2.0), (0.6, 0.5, 2.0)], gt)
        self.assertGreater(close, far)


def _write_det_fixture(root: Path) -> None:
    """Two images: image 0 has one cat-0 GT box matched by a good + a spurious prediction;
    image 1 has one cat-1 GT box matched by one good prediction."""
    writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
    writer.declare_numeric("capture.det.boxes", shape=(4,), dtype="float32")
    writer.declare_numeric("capture.det.category", shape=(1,), dtype="int64")
    writer.declare_numeric("capture.det.score", shape=(1,), dtype="float32")
    writer.declare_numeric("capture.det.sample_idx", shape=(1,), dtype="int64")
    writer.declare_numeric("batch.bboxes", shape=(4,), dtype="float32")
    writer.declare_numeric("batch.cls", shape=(1,), dtype="int64")
    writer.declare_numeric("batch.sample_idx", shape=(1,), dtype="int64")

    predictions = [
        (0, [0.5, 0.5, 0.2, 0.2], 0, 0.9),  # exact match on image 0's GT box -> TP
        (0, [0.1, 0.1, 0.05, 0.05], 0, 0.5),  # no overlap -> FP
        (1, [0.3, 0.3, 0.1, 0.1], 1, 0.8),  # exact match on image 1's GT box -> TP
    ]
    for sample_idx, box, cat, score in predictions:
        writer.append_numeric("capture.det.boxes", [box])
        writer.append_numeric("capture.det.category", [[cat]])
        writer.append_numeric("capture.det.score", [[score]])
        writer.append_numeric("capture.det.sample_idx", [[sample_idx]])

    ground_truth = [
        (0, [0.5, 0.5, 0.2, 0.2], 0),
        (1, [0.3, 0.3, 0.1, 0.1], 1),
    ]
    for sample_idx, box, cat in ground_truth:
        writer.append_numeric("batch.bboxes", [box])
        writer.append_numeric("batch.cls", [[cat]])
        writer.append_numeric("batch.sample_idx", [[sample_idx]])

    writer.declare_route("det", boxes="capture.det.boxes", category="capture.det.category", score="capture.det.score", sample_idx="capture.det.sample_idx")
    writer.close()


class InstanceMatchTests(unittest.TestCase):
    def test_matches_by_category_and_image_ranks_by_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_det_fixture(root)
            source = RunEvalSource(root, split="test")

            metric = InstanceMatch(similarity=IouXyxy(), overlap_thresholds=[0.5], **_DET_KEYS)
            result = metric.from_source(source)

            self.assertEqual(result["n_predictions"], 3)
            self.assertEqual(result["n_ground_truth"], 2)
            rows = result["rows"]
            # Ranked by score descending: 0.9 (cat0, TP), 0.8 (cat1, TP), 0.5 (cat0, FP).
            self.assertEqual([round(r["score"], 2) for r in rows], [0.9, 0.8, 0.5])
            self.assertEqual([r["is_true_positive"] for r in rows], [True, True, False])
            self.assertEqual([r["category"] for r in rows], [0, 1, 0])
            self.assertEqual(result["ground_truth"], {0: 1, 1: 1})

    def test_one_matching_pass_per_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_det_fixture(root)
            source = RunEvalSource(root, split="test")
            result = InstanceMatch(similarity=IouXyxy(), overlap_thresholds=[0.5, 0.9], **_DET_KEYS).from_source(source)
            self.assertEqual(sorted({r["threshold"] for r in result["rows"]}), [0.5, 0.9])
            self.assertEqual(len(result["rows"]), 6)
            self.assertEqual(result["n_predictions"], 3)

    def test_a_gt_instance_can_only_be_claimed_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
            writer.declare_numeric("capture.det.boxes", shape=(4,), dtype="float32")
            writer.declare_numeric("capture.det.category", shape=(1,), dtype="int64")
            writer.declare_numeric("capture.det.score", shape=(1,), dtype="float32")
            writer.declare_numeric("capture.det.sample_idx", shape=(1,), dtype="int64")
            writer.declare_numeric("batch.bboxes", shape=(4,), dtype="float32")
            writer.declare_numeric("batch.cls", shape=(1,), dtype="int64")
            writer.declare_numeric("batch.sample_idx", shape=(1,), dtype="int64")
            # Two predictions on the same GT box; only the higher-scoring one should win it.
            for box, score in ([0.5, 0.5, 0.2, 0.2], 0.9), ([0.5, 0.5, 0.2, 0.2], 0.7):
                writer.append_numeric("capture.det.boxes", [box])
                writer.append_numeric("capture.det.category", [[0]])
                writer.append_numeric("capture.det.score", [[score]])
                writer.append_numeric("capture.det.sample_idx", [[0]])
            writer.append_numeric("batch.bboxes", [[0.5, 0.5, 0.2, 0.2]])
            writer.append_numeric("batch.cls", [[0]])
            writer.append_numeric("batch.sample_idx", [[0]])
            writer.declare_route("det", boxes="capture.det.boxes", category="capture.det.category", score="capture.det.score", sample_idx="capture.det.sample_idx")
            writer.close()

            source = RunEvalSource(root, split="test")
            result = InstanceMatch(similarity=IouXyxy(), overlap_thresholds=[0.5], **_DET_KEYS).from_source(source)
            self.assertEqual([r["is_true_positive"] for r in result["rows"]], [True, False])

    def test_feeds_ranked_average_precision_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_det_fixture(root)
            source = RunEvalSource(root, split="test")

            matched = InstanceMatch(similarity=IouXyxy(), overlap_thresholds=[0.5], **_DET_KEYS).from_source(source)
            ap_result = RankedAveragePrecision().from_source(source, matched=matched)

            # category 0: ranked [TP(0.9), FP(0.5)] -> recall reaches 1.0 immediately at rank 1.
            # category 1: ranked [TP(0.8)] -> recall reaches 1.0 immediately.
            # Both categories' first-rank precision is 1.0, so mean AP should be exactly 1.0
            # under 101-point interpolation (every recall point's max-precision-to-the-right is 1.0).
            self.assertAlmostEqual(ap_result["mean_average_precision"], 1.0, places=6)
            categories = {row["category"] for row in ap_result["per_category"]}
            self.assertEqual(categories, {"0", "1"})


if __name__ == "__main__":
    unittest.main()
