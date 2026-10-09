"""Parameter-sync strategies for ``bind: {mode: frozen}`` nodes.

A ``frozen`` bind node holds a separate, non-trainable copy of a source node's module
whose weights are slaved to the source by one of these strategies (driven intrinsically by
``TeiaNetModule`` each training batch — no external callback). The set is open: a
strategy is an ordinary ``_target_`` plugin, so packs/users can supply their own (scheduled
τ, soft-then-hard hybrids, distillation, …).

See teia:core/module/bind.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class SyncStrategy(ABC):
    """How a ``frozen`` clone tracks its source module's parameters over training steps."""

    @abstractmethod
    def sync(self, target: nn.Module, source: nn.Module, *, global_step: int) -> None:
        """Update ``target``'s params/buffers from ``source`` (in place, no grad)."""


class HardSync(SyncStrategy):
    """Wholesale copy ``target ← source`` every ``every`` global steps (hard-updated target net)."""

    def __init__(self, every: int = 1) -> None:
        self.every = max(int(every), 1)

    def sync(self, target: nn.Module, source: nn.Module, *, global_step: int) -> None:
        if int(global_step) % self.every == 0:
            target.load_state_dict(source.state_dict())


class EmaSync(SyncStrategy):
    """Polyak / exponential-moving-average track ``target ← (1−τ)·target + τ·source`` each step.

    Used for soft target networks and EMA teachers.
    Buffers (e.g. BatchNorm running stats) are copied, not averaged — the standard convention.
    """

    def __init__(self, tau: float = 0.005) -> None:
        if not 0.0 < float(tau) <= 1.0:
            raise ValueError(f"EmaSync.tau must be in (0, 1], got {tau!r}.")
        self.tau = float(tau)

    def sync(self, target: nn.Module, source: nn.Module, *, global_step: int) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.lerp_(sp.detach(), self.tau)
            for tb, sb in zip(target.buffers(), source.buffers()):
                tb.copy_(sb)
