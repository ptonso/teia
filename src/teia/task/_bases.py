"""Shared task-contract task contract.

Source: common knowledge

Description:
  Input sets several task contracts share: an RGB image batch, the tabular type slots, and the interactive-regime transition fields. See teia:base/task.md.
"""

from __future__ import annotations

from teia.base.task import Blob, Dim, Names, Ragged, TaskContract, Tensor

IMAGE = {"image": Tensor("B 3 H W", "float32")}
CLASSES = {"num_classes": Dim(), "class_names": Names("num_classes")}
TABULAR = {
    name: Tensor("B *")
    for name in ("numerical", "categorical", "count", "temporal", "boolean", "numerical_mask", "temporal_mask")
}
DET_TARGETS = {
    "bboxes": Ragged("N 4", index="batch_idx", target=True),
    "cls": Ragged("N *", index="batch_idx", target=True),
    "batch_idx": Ragged("N *", index="batch_idx", target=True),
}
DET_CAPTURE = {
    "det.boxes": Ragged("N 4", index="det.sample_idx"),
    "det.score": Ragged("N", index="det.sample_idx"),
    "det.category": Ragged("N", index="det.sample_idx"),
    "det.sample_idx": Ragged("N", index="det.sample_idx"),
}
TRANSITION = {
    name: Tensor("B *")
    for name in ("obs_state", "action", "reward", "terminated", "truncated", "mask", "episode_start", "episode_id")
}
ON_POLICY = {name: Tensor("B *") for name in ("log_prob", "value", "advantage", "return_")}
REPLAY = {name: Tensor("B *") for name in ("next_obs_state", "weight", "index", "n_step_discount")}


class SingleLabel(TaskContract):
    """Single-label classification boundary B: class scores and the argmax label."""

    meta = CLASSES
    capture = {"cls.scores": Tensor("B num_classes", "float32"), "cls.label": Tensor("B")}


class MultiLabel(TaskContract):
    """Multi-label classification boundary B: per-class scores and thresholded multi-hot labels."""

    meta = CLASSES
    capture = {"cls.scores": Tensor("B num_classes", "float32"), "cls.labels": Tensor("B num_classes")}
