"""``Parameter``: a learned tensor emitted at every forward.

Source: common knowledge

Description:
  Holds one trainable tensor filled with ``init`` and emits it, ignoring its input. The declared out shape sets the
  tensor shape (none declared means a scalar). A state-independent log standard deviation is ``[action_dim]``, a SAC
  log-temperature is a scalar.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from teia.base.net import TeiaNode


class Parameter(TeiaNode):
    """Trainable tensor of the declared out shape, initialised to ``init``."""

    def build_module(self, init: float = 0.0, **_: Any) -> None:
        shape = tuple(int(d) for d in self.out_shape) if self.out_shape else ()
        self.value = nn.Parameter(torch.full(shape, float(init)))

    def forward(self, _ignored: Tensor | None = None) -> Tensor:
        return self.value


__all__ = ["Parameter"]
