"""
Gaussian Negative Log-Likelihood Loss.

Source: common knowledge

Description:
  Optimizes a regression mean by minimizing the normal negative log-likelihood under a fixed unit
  variance, which reduces to a scaled squared error plus a constant. It consumes a predicted mean
  and continuous targets.

Adaptations:
  - Variance is fixed at 1 rather than predicted, so this is not the heteroscedastic form: there
    is no second output head and no learned spread. The ``0.5*log(2*pi)`` term is kept so the
    reported value is a true NLL, though it contributes no gradient.
"""


from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class GaussianNLLLoss(BaseLoss):
    """Gaussian negative log-likelihood over a predicted mean at fixed unit variance."""
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # Fixed unit variance: NLL = 0.5 * (mean - target)^2 + 0.5*log(2π).
        return 0.5 * F.mse_loss(logits, target.float()) + _HALF_LOG_2PI
