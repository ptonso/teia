"""``PolicyAdapter`` — the Runner's only window into the bound policy module (contract).

The adapter turns one act batch into the captured atoms the task contract names (``{"policy.action": ...}``)
without the loop reaching into module internals. It is the **policy-query seam**: swapping the action
distribution changes only the module preset, never the loop.

``PolicyAdapter`` is an ABC so non-feedforward policies (recurrent R2D2, model-based Dreamer) can supply their
own ``act`` and carry per-env latent state, cleared in ``reset``. The canonical stateless implementation
``FeedforwardPolicyAdapter`` is plumbing and lives in ``teia.core.datamodule.interactive.policy``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class PolicyAdapter(ABC):
    """The Runner's window into the policy: ``act`` one batch, ``reset`` finished environments."""

    @property
    @abstractmethod
    def module(self) -> Any:
        """The bound policy module (the live Lightning module during fit)."""

    @abstractmethod
    def act(
        self, act_batch: Any, *, deterministic: bool = False, generator: torch.Generator | None = None
    ) -> dict[str, Tensor]:
        """Captured atoms for ``act_batch``, keyed by the contract's capture names."""

    def reset(self, boundary: Tensor) -> None:
        """Clear per-env latent state where ``boundary`` is true. No-op for feedforward."""
        return None


__all__ = ["PolicyAdapter"]
