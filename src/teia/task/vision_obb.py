"""Oriented-box detection task contract.

Source: common knowledge

Description:
  Detection with rotated boxes (fractional ``xywhr``). See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import CLASSES, DET_CAPTURE, DET_TARGETS, IMAGE, Ragged, TaskContract


class VisionObb(TaskContract):
    """``batch.image`` → ``capture.det.*`` with 5-column boxes; ground truth ``batch.bboxes`` is ``xywhr``."""

    meta = CLASSES
    batch = {**IMAGE, **DET_TARGETS, "bboxes": Ragged("N 5", index="batch_idx", target=True)}
    capture = {**DET_CAPTURE, "det.boxes": Ragged("N 5", index="det.sample_idx")}
