"""Irreducible interactive/interactive-regime record types.

These NamedTuples carry no natural non-interactive name (unlike the retired pillar ABCs) — they
describe environment-interaction shapes, not data-graph node contracts. The runtime batch
type is now graph-declared via ``compose_batch`` (``teia.base.fields``), not derived here
from a probed ``EnvSpec``.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

from torch import Tensor


class EnvSpec(NamedTuple):
    """Datamodule-owned runtime metadata derived from the environment spaces.

    Probed lazily on first access to ``EnvReader.env_spec`` (the executor triggers this
    via ``runtime_dims()`` during ``ensure_task_metadata()``).
    """

    action_space: str
    action_shape: tuple[int, ...]
    action_low: Tensor | None
    action_high: Tensor | None
    observation_shapes: dict[str, tuple[int, ...]]
    observation_dtypes: dict[str, str]
    num_envs: int
    supports_render: bool


class EnvStepResult(NamedTuple):
    """Per-step record produced by ``BaseEnvIO.step`` — environment fields only.

    IO has no access to policy outputs (action sampling/value prediction happen after
    the step is requested), so it returns only environment-derived fields. The
    collection loop upgrades this into a ``TransitionRecord``.
    """

    obs: Mapping[str, Tensor]
    next_obs: Mapping[str, Tensor]
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_id: Tensor | None = None


class TransitionRecord(NamedTuple):
    """Canonical collector-owned record consumed by the buffer.

    ``EnvStepResult`` plus the policy-derived fields the collection loop attaches:
    ``action``/``env_action`` (sampled) and ``log_prob``/``value`` (read from the
    cached forward). The ``obs``/``next_obs`` mappings are an internal IO/buffer
    convenience; they are flattened into the stable ``obs_<key>`` / ``next_obs_<key>``
    fields before any batch crosses the nn boundary.
    """

    obs: Mapping[str, Tensor]
    next_obs: Mapping[str, Tensor]
    action: Tensor
    env_action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    log_prob: Tensor | None = None
    value: Tensor | None = None
    episode_id: Tensor | None = None


class ActionSample(NamedTuple):
    """Result of an interactive activation's ``sample()`` extension.

    Deliberately omits ``value``: the bootstrap value is a property of the value head,
    not the action distribution, so the collection loop reads it from ``pred.value``
    via ``PolicyAdapter.value(obs)`` and stores it on the ``TransitionRecord``
    separately.
    """

    action: Tensor
    env_action: Tensor
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
