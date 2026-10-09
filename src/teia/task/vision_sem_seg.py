"""Semantic segmentation task contract.

Source: common knowledge

Description:
  An RGB image in; a class-index mask out, both masks stored as PNG blobs. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import CLASSES, IMAGE, Blob, TaskContract


class VisionSemSeg(TaskContract):
    """``batch.image`` → ``capture.seg.mask``, ground truth ``batch.masks``."""

    meta = CLASSES
    batch = {**IMAGE, "masks": Blob("png", target=True)}
    capture = {"seg.mask": Blob("png")}
