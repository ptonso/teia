"""
Split Node.

Source: common knowledge

Description:
  Fans a single feature tensor out to N identical outputs — the generic multi-task tap used when
  several downstream heads each need their own copy of the same upstream feature. Carries no
  parameters; each downstream node applies its own transform.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from teia.base.net import TeiaNode


class Split(TeiaNode):
    """``in: feat.x``  ``out: [feat.a, feat.b, …]``"""

    def build_module(self, n_outputs: int = 2, **kwargs: Any) -> None:
        self.n_outputs = int(n_outputs)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(x for _ in range(self.n_outputs))
