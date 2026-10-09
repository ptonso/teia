"""SemanticConfusion parity test: pixel-level confusion
derivation, hand-verified against a small 2x2 mask pair rather than the legacy record-shaped
``sem_seg.py::eval_objective`` (which takes composite ``PredictionRecord``s, not raw arrays) —
the formulas are ported line-for-line from that function, so a hand-computed fixture is the more
direct correctness check."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from teia.core.eval.runner import RunEvalSource
from teia.core.capture.store import CaptureWriter
from teia.node.eval.metric.aligned.semantic_confusion import SemanticConfusion


def _write_mask(path: Path, array: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


class SemanticConfusionTests(unittest.TestCase):
    def test_pixel_confusion_matches_hand_computed_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "masks" / "gt.png"
            pred_path = root / "masks" / "pred.png"
            _write_mask(gt_path, [[0, 0], [1, 1]])
            _write_mask(pred_path, [[0, 1], [1, 1]])  # one pixel wrong: gt=0 predicted=1

            writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
            writer.declare_rows("capture.sem-seg.mask_path")
            writer.declare_rows("batch.mask_path")
            writer.append_rows("capture.sem-seg.mask_path", [str(pred_path)])
            writer.append_rows("batch.mask_path", [str(gt_path)])
            writer.declare_route("sem-seg", mask_path="capture.sem-seg.mask_path")
            writer.close()

            source = RunEvalSource(root, split="test")
            metric = SemanticConfusion(pred="capture.sem-seg.mask_path", gt="batch.mask_path", ignore_index=255, worst_k=5)
            result = metric.from_source(source, class_names=["bg", "fg"])

            self.assertAlmostEqual(result["pixel_accuracy"], 0.75, places=6)
            self.assertAlmostEqual(result["mean_iou"], 0.583333, places=5)
            self.assertAlmostEqual(result["mean_dice"], 0.733333, places=5)
            self.assertAlmostEqual(result["mean_pixel_accuracy"], 0.75, places=6)
            self.assertAlmostEqual(result["frequency_weighted_iou"], 0.583333, places=5)
            self.assertEqual(result["confusion_matrix"], [[1.0, 1.0], [0.0, 2.0]])

            class_rows = {row["class_name"]: row for row in result["class_rows"]}
            self.assertAlmostEqual(class_rows["bg"]["iou"], 0.5, places=6)
            self.assertAlmostEqual(class_rows["fg"]["iou"], 2.0 / 3.0, places=6)

    def test_ignore_index_pixels_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "masks" / "gt.png"
            pred_path = root / "masks" / "pred.png"
            _write_mask(gt_path, [[0, 255], [1, 1]])  # top-right pixel is void
            _write_mask(pred_path, [[0, 0], [1, 1]])  # would be "wrong" if not ignored

            writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
            writer.declare_rows("capture.sem-seg.mask_path")
            writer.declare_rows("batch.mask_path")
            writer.append_rows("capture.sem-seg.mask_path", [str(pred_path)])
            writer.append_rows("batch.mask_path", [str(gt_path)])
            writer.declare_route("sem-seg", mask_path="capture.sem-seg.mask_path")
            writer.close()

            source = RunEvalSource(root, split="test")
            result = SemanticConfusion(pred="capture.sem-seg.mask_path", gt="batch.mask_path", ignore_index=255).from_source(source, class_names=["bg", "fg"])

            self.assertAlmostEqual(result["pixel_accuracy"], 1.0, places=6)
            self.assertAlmostEqual(result["void_pixel_rate"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
