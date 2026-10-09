"""
Vision Cross-Entropy Loss.

Source: common knowledge

Description:
  Wraps cross-entropy for semantic segmentation logits, resizing logits to target mask resolution before applying the categorical negative log-likelihood.
"""

from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss


class SegCrossEntropyLoss(BaseLoss):
    """CE loss for semantic segmentation; resizes logits to match mask size."""

    def __init__(
        self,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        self.class_weights = class_weights

    def forward(self, logits: Tensor, masks: Tensor) -> Tensor:
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(
            logits, masks.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
