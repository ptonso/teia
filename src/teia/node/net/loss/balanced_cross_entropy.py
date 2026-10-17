"""Balanced cross-entropy.

Source: common knowledge

Description:
  Cross-entropy averaged over the classes present in the batch, each class weighted by its own mean loss
  instead of by its number of samples. With class counts taken from a large batch it equals the inverse-frequency
  weighted cross-entropy, so training on the natural class mix keeps a balanced prior without any class
  weights in the config. A class absent from the batch contributes nothing.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss


class BalancedCrossEntropyLoss(BaseLoss):
    """Mean over the batch's classes of the per-class mean cross-entropy."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        per_sample = F.cross_entropy(logits, target.long(), reduction="none", label_smoothing=self.label_smoothing)
        one_hot = F.one_hot(target.long(), logits.shape[-1]).to(per_sample.dtype)
        counts = one_hot.sum(0)
        class_means = (one_hot * per_sample[:, None]).sum(0) / counts.clamp(min=1)
        present = (counts > 0).to(per_sample.dtype)
        return (class_means * present).sum() / present.sum()
