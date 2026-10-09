"""Instance segmentation task contract.

Source: common knowledge

Description:
  Detection plus one fractional polygon per instance. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import CLASSES, DET_CAPTURE, DET_TARGETS, IMAGE, Ragged, TaskContract, Tensor


class VisionInstSeg(TaskContract):
    """``batch.image`` → ``capture.det.*`` + ``capture.det.polys``; ground truth adds ``batch.polys``."""

    meta = CLASSES
    batch = {
        **IMAGE,
        **DET_TARGETS,
        "masks": Tensor("B *"),
        "sem_masks": Tensor("B *"),
        "polys": Ragged("N * 2", index="batch_idx", target=True),
    }
    capture = {**DET_CAPTURE, "det.polys": Ragged("N * 2", index="det.sample_idx")}
