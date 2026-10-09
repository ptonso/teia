"""
Sigmoid Tensor Activation.

Source: common knowledge

Description:
  Converts logits to independent sigmoid probabilities for binary or multilabel prediction routes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseActivation
from teia.node.net._activation_utils import decode_multilabel


class SigmoidActivation(BaseActivation):
    """Multi-label / binary classification: per-class sigmoid probabilities."""


    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        return {"probs": torch.sigmoid(logits)}

    @staticmethod
    def kernel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
        return decode_multilabel(activated, ctx)
