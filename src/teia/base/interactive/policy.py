"""``PolicyAdapter`` — the Runner's only window into the bound policy module (contract).

The adapter turns the generic node graph (``forward -> pred.*``) into the two operations the
collection loop needs — ``act`` (sample an action) and ``value`` (bootstrap) — without the
loop ever reaching into module internals. It is the **policy-query seam** (taxonomy axis E):
swapping the action distribution (Gaussian / epsilon-greedy / squashed-Gaussian)
changes only the module preset, never the loop.

``PolicyAdapter`` is an ABC so non-feedforward policies (recurrent R2D2, model-based Dreamer)
can supply their own ``act``/``value`` and carry per-env latent state. The canonical stateless
implementation ``FeedforwardPolicyAdapter`` is plumbing and lives in
``teia.core.datamodule.interactive.policy``.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from teia.base.interactive.batch import ActionSample


class PolicyAdapter(ABC):
    """The Runner's window into the policy: ``act`` (sample) and ``value`` (bootstrap).

    Stateless for the feedforward family; stateful (per-env latent state) for
    recurrent / model-based adapters, which override ``reset`` to clear finished envs.
    """

    @property
    @abstractmethod
    def module(self) -> Any:
        """The bound policy module (the live Lightning module during fit)."""

    @abstractmethod
    def act(
        self,
        obs_batch: Any,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> ActionSample:
        """Sample one action for ``obs_batch``; recurrent adapters advance/store state."""

    @abstractmethod
    def value(self, obs_batch: Any) -> Tensor:
        """Bootstrap value for ``obs_batch`` (off-policy adapters may not use it)."""

    def reset(self, done_mask: Tensor | None = None) -> None:
        """Reset per-env latent state where ``done_mask`` is true. No-op for feedforward."""
        return None


__all__ = ["PolicyAdapter"]
