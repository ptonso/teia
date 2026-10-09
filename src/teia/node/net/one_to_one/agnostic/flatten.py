"""
Flatten Adapter Node.

Source: common knowledge

Description:
  Flatten input tensor to a vector while preserving the batch dimension.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode



class Flatten(TeiaNode):
    """Flatten all non-batch dims to a vector: ``[B, *dims] → [B, prod(dims)]``.

    A generic shape adapter (e.g. feeding an image tensor into an :class:`MLP`). Carries no
    parameters; ``in_shape``/``out_shape`` are injected by the pipeline orchestrator.
    """

    def build_module(self) -> None:
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.flatten(x)
