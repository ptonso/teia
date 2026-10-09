"""
Identity Tensor Activation.

Source: common knowledge

Description:
  Exposes logits directly as scalar predictions for regression-style routes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from teia.base.net import BaseActivation
from teia.node.net._activation_utils import decode_regression


class IdentityActivation(BaseActivation):
    """Regression route: expose logits as the prediction value."""


    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        return {"value": logits}

    @staticmethod
    def kernel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
        return decode_regression(activated, ctx)
