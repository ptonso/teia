"""
Mean Absolute Error Loss.

Source: common knowledge

Description:
  Wraps PyTorch L1 loss as an atomic Teia loss node.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss

class MAELoss(BaseLoss):
    """Mean absolute error objective E[|pred - target|]."""
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(pred, target.float())
