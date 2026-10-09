"""
Binary Cross-Entropy Loss.

Source: common knowledge

Description:
  Wraps PyTorch binary cross-entropy with logits as an atomic Teia loss node.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss

class BCELoss(BaseLoss):
    """Binary cross-entropy objective, -y log p - (1-y) log(1-p)."""
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, target.float())
