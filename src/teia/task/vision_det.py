"""Object detection task contract.

Source: common knowledge

Description:
  An RGB image in; axis-aligned boxes (fractional ``cxcywh``) with class and score out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import CLASSES, DET_CAPTURE, DET_TARGETS, IMAGE, TaskContract


class VisionDet(TaskContract):
    """``batch.image`` → ``capture.det.*``, ground truth ``batch.{bboxes,cls,batch_idx}``."""

    meta = CLASSES
    batch = {**IMAGE, **DET_TARGETS}
    capture = DET_CAPTURE
