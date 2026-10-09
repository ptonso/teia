"""
Cross-Entropy Loss.

Source: common knowledge

Description:
  Wraps PyTorch categorical cross-entropy as an atomic Teia loss node.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss

class CrossEntropyLoss(BaseLoss):
    """Categorical cross-entropy objective over logits and class targets."""
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target.long(), label_smoothing=self.label_smoothing)

