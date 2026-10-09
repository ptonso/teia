"""
Mean Squared Error Loss.

Source: common knowledge

Description:
  Wraps PyTorch mean-squared error as an atomic Teia loss node.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss

class MSELoss(BaseLoss):
    """Mean squared error objective E[(pred - target)^2]."""
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target.float())
