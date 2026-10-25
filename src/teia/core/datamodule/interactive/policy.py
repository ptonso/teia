"""``FeedforwardPolicyAdapter`` — the canonical stateless ``PolicyAdapter`` (plumbing).

The ``PolicyAdapter`` contract lives in ``teia.base.interactive.policy``. Acting is the module's capture slice
(``TeiaNetModule.act_forward``): one pruned forward under ``no_grad``, never the full network.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from teia.base.interactive.policy import PolicyAdapter


class FeedforwardPolicyAdapter(PolicyAdapter):
    def __init__(self, module: Any) -> None:
        self._module = module

    @property
    def module(self) -> Any:
        return self._module

    def act(
        self, act_batch: Any, *, deterministic: bool = False, generator: torch.Generator | None = None
    ) -> dict[str, Tensor]:
        return self._module.act_forward(act_batch, deterministic=deterministic, generator=generator)
