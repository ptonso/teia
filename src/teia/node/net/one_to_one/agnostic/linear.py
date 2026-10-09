"""
Linear Projection Head.

Source: common knowledge

Description:
  Maps a flattened feature tensor to a configured output width with a single affine projection. This is the standard linear readout used by many Teia presets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode


class LinearHead(TeiaNode):
    """Project vector features to the declared output shape."""

    def build_module(self, **kwargs: Any) -> None:
        if self.out_shape is None:
            raise ValueError("LinearHead requires out_shape from node.out_key.")
        if not isinstance(self.in_shape, Sequence) or len(self.in_shape) != 1:
            raise ValueError(f"LinearHead requires 1-D in_shape, got {self.in_shape!r}.")
        if not isinstance(self.out_shape, Sequence) or isinstance(self.out_shape, (str, bytes)):
            raise ValueError(f"LinearHead requires a single output shape, got {self.out_shape!r}.")
        self._out_shape = tuple(int(dim) for dim in self.out_shape)
        self.linear = nn.Linear(int(self.in_shape[0]), int(math.prod(self._out_shape)))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).reshape(x.shape[0], *self._out_shape)
