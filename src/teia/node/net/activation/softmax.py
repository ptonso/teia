"""
Softmax Tensor Activation.

Source: common knowledge

Description:
  Converts class logits to softmax probabilities and an argmax class label.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseActivation
from teia.node.net._activation_utils import decode_singlelabel


class SoftmaxActivation(BaseActivation):
    """Single-label classification: softmax probabilities + argmax label."""


    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        probs = torch.softmax(logits, dim=-1)
        return {"probs": probs, "label": probs.argmax(dim=-1)}

    @staticmethod
    def kernel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
        return decode_singlelabel(activated, ctx)
