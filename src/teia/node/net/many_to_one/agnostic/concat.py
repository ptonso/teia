"""
Concatenation Fusion.

Source: common knowledge

Description:
  Concatenates N feature tensors along a fixed axis, optionally flattening the result into a
  dense ``[B, D]`` vector. The generic fusion half of a split: a fusion node followed by a plain dense body, instead of
  one node that both concatenates and transforms.
"""


from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from teia.base.net import TeiaNode


class ConcatFusion(TeiaNode):
    """``in: [feat.<a>, feat.<b>, …]``  ``out: feat.<fused>``

    Concatenates N inputs along ``axis``. With ``flatten: true``, collapses every axis after the
    batch axis, producing ``[B, D]`` — the form a dense body (e.g. ``one_to_one/agnostic/mlp.py``)
    expects.
    """

    def build_module(self, axis: int = 1, flatten: bool = False, **kwargs: Any) -> None:
        self.axis = int(axis)
        self.flatten = bool(flatten)

    def forward(self, *feats: Tensor) -> Tensor:
        out = torch.cat(feats, dim=self.axis)
        return out.flatten(1) if self.flatten else out
