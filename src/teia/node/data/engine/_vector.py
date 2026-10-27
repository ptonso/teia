"""``VectorEngine``: the shared plumbing of the vector-simulator engines.

A concrete engine builds the simulator, declares its signals (``_catalog``), and implements one reset (``_do_reset``)
and one step (``_do_step``) that return a dict of signal tensors with the environment axis first. This base validates
the recipe's signals at ``open``, packs the requested ones in ``out`` order, counts ``episode_step`` for the
simulators that keep no clock of their own, and fails fast on an action outside the simulator's finite bounds. It
never rescales or clips an action: the network owns the range.

Source: common knowledge

Description:
  Shared base of the vector engines: signal validation and packing, episode clock and action bound check.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from teia.base.data import Engine, EngineSpec

STATE, OUTCOME = "state", "outcome"
BOUND_TOLERANCE = 1e-5


class ActionSpace(NamedTuple):
    """``low``/``high`` are the simulator's bounds (tensors, possibly infinite) for a continuous action; a discrete
    action sets ``num_actions`` instead."""

    shape: tuple[int, ...]
    dtype: str
    low: Tensor | None = None
    high: Tensor | None = None
    num_actions: int | None = None


def to_tensor(array: Any) -> Tensor:
    """numpy (or tensor) to tensor: floats become float32, ints int64 (uint8 stays), bool stays."""
    if isinstance(array, Tensor):
        return array
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        return torch.as_tensor(arr, dtype=torch.float32)
    if arr.dtype == np.uint8 or np.issubdtype(arr.dtype, np.bool_):
        return torch.as_tensor(arr)
    return torch.as_tensor(arr, dtype=torch.int64)


def dtype_name(tensor: Tensor) -> str:
    """The bare dtype name (``float32``) of a tensor."""
    return str(tensor.dtype).rsplit(".", 1)[-1]


def world_id(data_root: str | None, scheme: str) -> str:
    """The id of ``data_root`` when it reads ``<scheme>:<id>``; any other value fails."""
    prefix = f"{scheme}:"
    if not data_root or not str(data_root).startswith(prefix):
        raise ValueError(f"this engine takes `data_root={prefix}<id>`; got {data_root!r}")
    return str(data_root)[len(prefix) :]


class VectorEngine(Engine):
    """Engine over a vector simulator; subclasses implement the simulator hooks."""

    def __init__(self, *, num_envs: int, seed: int = 0) -> None:
        self.num_envs, self.seed = int(num_envs), int(seed)
        self._sim: Any = None
        self._ep_step: Tensor | None = None
        self._bounds: ActionSpace | None = None

    # -- simulator hooks ----------------------------------------------------------
    def _build(self, seed: int) -> Any:
        raise NotImplementedError

    def _close_sim(self) -> None:
        close = getattr(self._sim, "close", None)
        if callable(close):
            close()

    def _catalog(self) -> dict[str, tuple[tuple[int, ...], str, str]]:
        """``{signal: (shape without the environment axis, dtype name, role)}`` of the simulator's signals."""
        raise NotImplementedError

    def _action_space(self) -> ActionSpace:
        raise NotImplementedError

    def _do_reset(self, mask: Tensor | None) -> dict[str, Tensor]:
        """Reset every environment, or the masked ones; return every signal for all environments."""
        raise NotImplementedError

    def _do_step(self, action: Tensor) -> dict[str, Tensor]:
        """Advance every environment; return every signal at the end of the transition."""
        raise NotImplementedError

    # -- Engine -------------------------------------------------------------------
    def _all_signals(self) -> dict[str, tuple[tuple[int, ...], str, str]]:
        return {**self._catalog(), "episode_step": ((), "int64", STATE)}

    def open(self, seed: int) -> None:
        if self._sim is not None:
            return
        self._sim = self._build(int(seed))
        catalog = self._all_signals()
        unknown = [s for s in self.signals if s not in catalog]
        if unknown:
            raise ValueError(f"{type(self).__name__} has no signal(s) {unknown}; it provides {sorted(catalog)}")
        self._bounds = self._action_space()
        self._ep_step = torch.zeros(self.num_envs, dtype=torch.int64)

    def close(self) -> None:
        if self._sim is not None:
            self._close_sim()
            self._sim = None

    def spec(self) -> EngineSpec:
        self.open(self.seed)
        catalog = self._all_signals()
        action = self._bounds
        bounded = action.low is not None
        return EngineSpec(
            num_envs=self.num_envs,
            outputs=tuple(catalog[s] for s in self.signals),
            action_shape=action.shape,
            action_dtype=action.dtype,
            action_low=float(action.low.min()) if bounded else None,
            action_high=float(action.high.max()) if bounded else None,
            num_actions=action.num_actions,
        )

    def _pick(self, values: dict[str, Tensor]) -> tuple[Tensor, ...]:
        values = {"episode_step": self._ep_step.clone(), **values}
        return tuple(values[s] for s in self.signals)

    def _check_bounds(self, action: Tensor) -> None:
        low, high = self._bounds.low, self._bounds.high
        if low is None:
            return
        low, high = low.to(action.device), high.to(action.device)
        outside = ((action < low - BOUND_TOLERANCE) | (action > high + BOUND_TOLERANCE)) & torch.isfinite(low) & torch.isfinite(high)
        if bool(outside.any()):
            raise ValueError(f"action outside the simulator's bounds [{float(low.min())}, {float(high.max())}]; the network's activation owns the range")

    def reset(self, mask: Tensor | None = None) -> tuple[Tensor, ...]:
        self.open(self.seed)
        values = self._do_reset(mask)
        self._ep_step = torch.zeros_like(self._ep_step) if mask is None else torch.where(mask.cpu(), torch.zeros_like(self._ep_step), self._ep_step)
        return self._pick(values)

    def step(self, action: Tensor) -> tuple[Tensor, ...]:
        self.open(self.seed)
        self._check_bounds(action)
        values = self._do_step(action)
        self._ep_step = self._ep_step + 1
        return self._pick(values)
