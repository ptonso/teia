"""Pose estimation task contract.

Source: common knowledge

Description:
  Detection plus ``kpt_shape`` fractional keypoints per instance. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import CLASSES, DET_CAPTURE, DET_TARGETS, IMAGE, Dim, Ragged, TaskContract


class VisionPose(TaskContract):
    """``batch.image`` → ``capture.det.*`` + ``capture.det.keypoints``; ground truth adds ``batch.keypoints``."""

    meta = {**CLASSES, "kpt_shape": Dim()}
    batch = {**IMAGE, **DET_TARGETS, "keypoints": Ragged("N kpt_shape", index="batch_idx", target=True)}
    capture = {**DET_CAPTURE, "det.keypoints": Ragged("N kpt_shape", index="det.sample_idx")}
