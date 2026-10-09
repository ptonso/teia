"""
Sum Fusion.

Source: common knowledge

Description:
  Element-wise sum of N same-shape feature tensors. The plain-addition sibling of
  ``concat.py``/``gated.py`` in the fusion family.
"""


from __future__ import annotations

from typing import Any

from torch import Tensor

from teia.base.net import TeiaNode


class SumFusion(TeiaNode):
    """``in: [feat.<a>, feat.<b>, …]`` (same shape)  ``out: feat.<fused>``

    Element-wise sum of N inputs of identical shape.
    """

    def build_module(self, **kwargs: Any) -> None:
        pass

    def forward(self, *feats: Tensor) -> Tensor:
        out = feats[0]
        for feat in feats[1:]:
            out = out + feat
        return out
