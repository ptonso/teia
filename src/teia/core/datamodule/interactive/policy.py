"""``FeedforwardPolicyAdapter`` — the canonical stateless ``PolicyAdapter`` (plumbing).

The ``PolicyAdapter`` contract (the Runner's only window into the bound policy module)
lives in ``teia.base.interactive.policy``. This module holds the stateless feedforward implementation
used by every policy and the default today.

The contract is **one env step = one model forward**: ``act`` runs ``forward``
once, caches the resulting ``pred.*``, samples via the primary activation's
``sample()`` extension, and ``value`` reads the cached ``pred.value`` without a
second forward.

"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from teia.base.interactive.batch import ActionSample
from teia.base.interactive.policy import PolicyAdapter


class FeedforwardPolicyAdapter(PolicyAdapter):
    """Stateless adapter over a single-forward, single-activation policy module."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self._cached_pred: Any | None = None
        self._cached_obs: Any | None = None

    @property
    def module(self) -> Any:
        return self._module

    def act(
        self,
        obs_batch: Any,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> ActionSample:
        """One forward (cached), then the primary activation's ``sample()``.

        Runs under ``torch.no_grad()`` so the stored tensors carry no graph.
        """
        with torch.no_grad():
            pred = self._module(obs_batch)
        self._cached_pred = pred
        self._cached_obs = obs_batch
        activation = self._primary_activation()
        logits = self._activation_logits(pred, activation)
        with torch.no_grad():
            return activation.sample(*logits, deterministic=deterministic, generator=generator)

    def value(self, obs_batch: Any) -> Tensor:
        """Bootstrap value from ``pred.value`` of the cached forward.

        Reuses the cached forward when ``obs_batch`` is the *same object* the current
        ``act`` cached; otherwise runs one fresh ``no_grad`` forward. This keeps the
        per-step contract at one forward (the step reuses ``act``'s cache) while the
        explicit post-rollout bootstrap — which passes a *different* observation — runs
        the single extra forward the buffer needs for ``last_value``.
        """
        if self._cached_pred is not None and obs_batch is self._cached_obs:
            pred = self._cached_pred
        else:
            with torch.no_grad():
                pred = self._module(obs_batch)
        value = self._read_pred(pred, "value")
        if value is None:
            # Off-policy presets carry no value head; replay bootstrapping reads
            # next-obs Q in the loss, not a stored value. Return a zero placeholder (unused).
            return torch.zeros(_leading_dim(obs_batch))
        return value

    # -- internals ----------------------------------------------------------------
    def _action_route(self) -> str:
        """The alias of the module's single action activation (the node exposing ``sample``, the ``ActionActivation`` protocol)."""
        names = [rec.name for rec in self._module._activation_records if callable(getattr(self._module._node(rec), "sample", None))]
        if len(names) != 1:
            raise RuntimeError(f"Policy module needs exactly one ActionActivation node; found {names}.")
        return names[0]

    def _primary_activation(self) -> Any:
        name = self._action_route()
        return self._module._node(next(rec for rec in self._module._activation_records if rec.name == name))

    @staticmethod
    def _read_pred(pred: Any, key: str) -> Any:
        if isinstance(pred, dict):
            return pred.get(f"pred.{key}", pred.get(key))
        return getattr(pred, key, None)

    def _activation_in_keys(self, activation: Any) -> list[str]:
        """The ``pred.*`` keys the primary activation consumes, in declared order.

        Read from the module's normalized activation records (the built activation node has no
        ``.node`` envelope — that is stripped at build). Falls back to ``activation.node.in_key``
        for lightweight test doubles, then to ``pred.policy``.
        """
        records = getattr(self._module, "_activation_records", None)
        name = self._action_route() if records else None
        if records:
            for rec in records:
                if getattr(rec, "name", None) == name:
                    return list(rec.in_key)
        raw = getattr(getattr(activation, "node", None), "in_key", None)
        if isinstance(raw, str):
            return [raw]
        return list(raw) if raw else ["pred.policy"]

    def _activation_logits(self, pred: Any, activation: Any) -> tuple[Tensor, ...]:
        """Gather the ``pred.*`` inputs the activation node consumes, in declared order."""
        logits: list[Tensor] = []
        for key in self._activation_in_keys(activation):
            short = str(key).split(".", 1)[-1]
            value = self._read_pred(pred, short)
            if value is not None:
                logits.append(value)
        return tuple(logits)


def _leading_dim(obs_batch: Any) -> int:
    """Batch size from the first tensor field of an obs batch (for a zero value placeholder)."""
    fields = getattr(obs_batch, "_fields", None)
    if fields:
        values = (getattr(obs_batch, f) for f in fields)
    elif hasattr(obs_batch, "__dict__"):
        values = vars(obs_batch).values()
    else:
        values = (obs_batch,)
    for value in values:
        if isinstance(value, Tensor):
            return int(value.shape[0])
    return 1
