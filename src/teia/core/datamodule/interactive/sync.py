"""Synchronous Runner — collection and training never overlap (one thread, two phases).

The on-policy / off-policy distinction is carried by the **stream-reshape node** (the buffer),
not the Runner: this single loop adapts to whichever reshape node is declared — or, if none is
declared, drives an internally-constructed :class:`~teia.core.datamodule.interactive.rollout.RolloutBuffer`
(the bufferless on-policy default; same GAE math either way).

* Rollout buffer (on-policy) — ``ready()`` fires after a full rollout;
  ``finalize_rollout()`` computes GAE off a post-rollout bootstrap value; ``sample_iter()``
  drains ``update_epochs × minibatches``; ``on_consumed()`` clears the buffer.
* Replay buffer (off-policy) — ``ready()`` fires after warmup/``train_frequency``;
  no finalize; ``sample_iter()`` yields ``updates_per_cycle`` sampled minibatches;
  ``on_consumed()`` is a no-op (the buffer persists). Replay buffers also need ``next_obs``
  stored per step (TD / n-step targets read it from ``batch.*``).

One env step = one policy forward (``PolicyAction`` caches the forward via its internal
``FeedforwardPolicyAdapter``). Per-env autoreset bookkeeping masks the spurious reset step
(``mask=0`` on the step immediately following a done, under next-step autoreset).

The item-stage subgraph (``TeiaDataModule._item_workspace``) — obs-normalize, ``PolicyAction`` —
is reused verbatim from the offline path: item-stage nodes don't know or care whether they're
invoked from ``DataLoader`` workers over a static index or from this inline generator loop. Batch
assembly (``TeiaDataModule._collate_fn``) is likewise reused verbatim.

"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import torch

from teia.core.datamodule.interactive._tensor_ops import detach_cpu, to_device
from teia.core.datamodule.interactive.runner_base import Runner


def _obs_seed(raw_obs: Mapping[str, Any], ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Explode a raw ``{obs_key: tensor}`` mapping into ``item.obs_<key>`` seed entries."""
    seed: dict[str, Any] = {f"item.obs_{key}": value for key, value in raw_obs.items()}
    seed.update({f"ctx.{key}": value for key, value in ctx.items()})
    return seed


def _set_training(dm: Any, training: bool) -> None:
    for rec in dm._item_records:
        node = dm._nodes[rec.name]
        if hasattr(node, "training"):
            node.training = training


def _normalize_only(dm: Any, raw_obs: Mapping[str, Any], *, training: bool) -> dict[str, Any]:
    """Run only the obs-normalize nodes (item-stage transforms whose out_key is obs-shaped) —
    used to normalize ``next_obs`` for off-policy buffers without a second policy forward."""
    workspace: dict[str, Any] = {f"item.obs_{key}": value for key, value in raw_obs.items()}
    touched: list[tuple[Any, bool]] = []
    try:
        for rec in dm._item_records:
            if rec.phase == "augment" or rec.kind != "transform":
                continue
            if not rec.out_key or not all(k.startswith("item.obs") for k in rec.out_key):
                continue
            node = dm._nodes[rec.name]
            if hasattr(node, "training"):
                touched.append((node, node.training))
                node.training = training
            if all(k in workspace for k in rec.in_key):
                result = node(*[workspace[k] for k in rec.in_key])
                if len(rec.out_key) == 1:
                    workspace[rec.out_key[0]] = result
                else:
                    for key, value in zip(rec.out_key, result):
                        workspace[key] = value
    finally:
        for node, prev in touched:
            node.training = prev
    return workspace


def _next_obs_fields(next_workspace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"item.next_obs_{key[len('item.obs_') :]}": value
        for key, value in next_workspace.items()
        if key.startswith("item.obs_")
    }


class SyncRunner(Runner):
    """Inline synchronous collect↔train loop; on/off-policy is the reshape node's choice.

    With no ``stream``-stage reshape node declared it runs the **bufferless** on-policy path:
    an internally-constructed ``RolloutBuffer`` sized from this runner's own
    ``rollout_steps``/``minibatch_size``/``update_epochs``/``gamma`` — the day-1 default.
    """

    def __init__(
        self,
        *,
        rollout_steps: int = 2048,
        minibatch_size: int = 64,
        update_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        **_: Any,
    ) -> None:
        self.rollout_steps = int(rollout_steps)
        self.minibatch_size = int(minibatch_size)
        self.update_epochs = int(update_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

    def iter_minibatches(self) -> Iterator[Any]:
        ctx = self.ctx
        reshape = ctx.stream_reshape
        if reshape is None:
            from teia.core.datamodule.interactive.rollout import RolloutBuffer

            reshape = RolloutBuffer(
                rollout_steps=self.rollout_steps,
                minibatch_size=self.minibatch_size,
                update_epochs=self.update_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        reshape.configure(num_envs=ctx.num_envs, generator=ctx.generator)
        yield from self._drive(reshape)

    def _drive(self, reshape: Any) -> Iterator[Any]:
        ctx = self.ctx
        dm, reader = ctx.dm, ctx.stream_reader
        reader.setup_stream()
        _set_training(dm, ctx.training)
        is_rollout = callable(getattr(reshape, "finalize_rollout", None))

        raw_obs = to_device(reader.reset(), ctx.device)
        prev_done = torch.zeros(ctx.num_envs, dtype=torch.bool)
        env_step = 0
        episode_log = ctx.episode_log if ctx.training else None
        ep_return = torch.zeros(ctx.num_envs)
        ep_length = torch.zeros(ctx.num_envs)

        while env_step < ctx.total_env_steps:
            workspace = dm._item_workspace(_obs_seed(raw_obs, ctx.ctx), augment=False)
            env_action = detach_cpu(workspace["item.env_action"])
            step = reader.step(env_action)

            mask = (~prev_done).to(torch.float32)
            episode_start = prev_done.to(torch.float32)
            done = (step.terminated | step.truncated).to(torch.bool)

            next_obs = to_device(step.next_obs, ctx.device)

            item: dict[str, Any] = {k: v for k, v in workspace.items() if k.startswith("item.")}
            item["item.reward"] = step.reward
            item["item.terminated"] = step.terminated
            item["item.truncated"] = step.truncated
            item["item.mask"] = mask
            item["item.episode_start"] = episode_start
            if not is_rollout:
                next_ws = _normalize_only(dm, next_obs, training=False)
                item.update(_next_obs_fields(next_ws))
            reshape.add(item)

            if episode_log is not None:
                ep_return += step.reward.detach().to(ep_return.dtype).reshape(-1) * mask
                ep_length += mask
                for i in torch.nonzero(done & (mask > 0)).reshape(-1).tolist():
                    episode_log.append((float(ep_return[i]), int(ep_length[i])))
                    ep_return[i] = 0.0
                    ep_length[i] = 0.0

            raw_obs = next_obs
            prev_done = done
            env_step += ctx.num_envs

            if reshape.ready():
                if is_rollout:
                    boot_ws = dm._item_workspace(_obs_seed(raw_obs, ctx.ctx), augment=False)
                    last_value = detach_cpu(boot_ws["item.value"]).reshape(-1)
                    reshape.finalize_rollout(last_value=last_value, last_done=prev_done.to(torch.float32))
                for minibatch_seeds in reshape.sample_iter():
                    yield dm._collate_fn(minibatch_seeds)
                reshape.on_consumed()
