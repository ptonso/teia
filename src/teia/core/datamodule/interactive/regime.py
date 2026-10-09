"""``InteractiveRegime`` — couples online collection to training for a live env source.

The regime is the interactive-specific orchestration layer ``TeiaDataModule`` delegates to when its data
graph carries a ``stage="stream"`` reader (``self._live``). Unlike the pre-cutover version, it
owns no pillar bundle of its own — every method drives the *generic* graph
(``dm._item_workspace``/``dm._collate_fn``/``dm._stream_reader_node()``/``dm._stream_reshape_node()``)
instead of a bespoke ``io``/``transform``/``buffer``/``collater`` tuple:

* ``train_iterable(module)`` — bind a ``RunnerContext`` (graph reader/reshape/ctx) and return the
  configured ``Runner``'s iterable (num_workers=0, live weights — one env step = one policy
  forward inside the item subgraph's ``PolicyAction`` node).
* ``eval_iterable(...)`` / ``evaluate(...)`` — deterministic capture-only eval episodes, driven
  the same way (item subgraph + collate_fn), reading the env directly off the stream reader's
  ``io`` backend for the one-off eval env (never the training ``train_env``).

See teia:core/datamodule.md (interactive regime).
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset

from teia.core.datamodule.interactive._tensor_ops import detach_cpu, to_device
from teia.core.datamodule.interactive.sync import _obs_seed


class InteractiveRegime:
    """Collect↔train coupling over a live graph (``dm`` is the owning ``TeiaDataModule``)."""

    def __init__(self, dm: Any) -> None:
        self._dm = dm

    # -- training ----------------------------------------------------------------
    def train_iterable(self, module: Any) -> IterableDataset:
        dm = self._dm
        dm.bind_context(policy=module)
        ctx = dm._runner_context(training=True)
        dm._runner.configure(ctx)
        return _RunnerIterable(dm._runner)

    # -- evaluation --------------------------------------------------------------
    def eval_iterable(self, module: Any, *, episodes: int, deterministic: bool, seed: int) -> IterableDataset:
        return _StreamEvalDataset(regime=self, module=module, episodes=episodes, deterministic=deterministic, seed=seed)

    def sample_batch(self) -> Any:
        """One collated eval-shaped batch, for ``batch_meta()``/export shape probing. Requires
        ``ctx.policy`` already bound (e.g. via ``bind_context``) since no trainer is attached."""
        dm = self._dm
        module = dm.resolve_context("policy")
        record = next(iter(self.eval_steps(module, episodes=1, deterministic=True, seed=0)))
        return dm._collate_fn([_eval_seed_item(record)])

    def evaluate(
        self, module: Any, *, episodes: int = 10, deterministic: bool = True, seed: int | None = None
    ) -> dict[str, list[float]]:
        returns: list[float] = []
        lengths: list[float] = []
        terminated_flags: list[float] = []
        for record in self.eval_steps(module, episodes=episodes, deterministic=deterministic, seed=seed):
            if record["episode_done"]:
                returns.append(record["episode_return"])
                lengths.append(record["episode_length"])
                terminated_flags.append(record["episode_terminated"])
        return {"return": returns, "length": lengths, "terminated": terminated_flags}

    def eval_steps(
        self, module: Any, *, episodes: int, deterministic: bool, seed: int | None
    ) -> Iterator[dict[str, Any]]:
        """Per-step eval generator shared by ``evaluate`` and the eval dataloader (capture-only)."""
        dm = self._dm
        dm.bind_context(policy=module)
        reader = dm._stream_reader_node()
        io = reader.io
        device = _policy_device(module)
        eval_seed = int(reader.seed if seed is None else seed)
        env = io.build_eval_env(eval_seed, reader.num_envs)
        ctx_values = dm._bind_stream_context()
        gen = torch.Generator()
        gen.manual_seed(eval_seed + 11)

        raw_obs = to_device(io.reset(env, seed=eval_seed), device)
        prev_done = torch.zeros(reader.num_envs, dtype=torch.bool)
        episode_id = torch.zeros(reader.num_envs, dtype=torch.long)
        ep_return = torch.zeros(reader.num_envs)
        ep_length = torch.zeros(reader.num_envs)
        completed = 0
        try:
            while completed < int(episodes):
                seed_ws = _obs_seed(raw_obs, ctx_values)
                workspace = dm._item_workspace(seed_ws, augment=False)
                norm_obs = {k: v for k, v in workspace.items() if k.startswith("item.obs_")}
                env_action = detach_cpu(workspace["item.env_action"])
                step = io.step(env, env_action)

                mask = (~prev_done).to(torch.float32)
                ep_return += step.reward.detach().to(ep_return.dtype).reshape(-1) * mask
                ep_length += mask
                done = (step.terminated | step.truncated).to(torch.bool)

                for env_i in range(reader.num_envs):
                    finished = bool(done[env_i]) and bool(mask[env_i] > 0)
                    yield {
                        "obs": {k[len("item.obs_") :]: v[env_i] for k, v in norm_obs.items()},
                        "action": detach_cpu(workspace["item.action"])[env_i],
                        "reward": step.reward[env_i],
                        "terminated": step.terminated[env_i],
                        "truncated": step.truncated[env_i],
                        "episode_id": int(episode_id[env_i]),
                        "mask": float(mask[env_i]),
                        "episode_done": finished,
                        "episode_return": float(ep_return[env_i]),
                        "episode_length": int(ep_length[env_i]),
                        "episode_terminated": float(bool(step.terminated[env_i])),
                    }
                    if finished:
                        completed += 1
                        episode_id[env_i] += 1
                        ep_return[env_i] = 0.0
                        ep_length[env_i] = 0.0

                raw_obs = to_device(step.next_obs, device)
                prev_done = done
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def batch_meta(self, dims: dict[str, Any], fields: set[str]) -> dict[str, tuple[int, ...]]:
        """Static shape inference from published ``runtime_dims()``, restricted to ``fields``
        (the datamodule's declared collate fields) — no live policy required (unlike the
        item-subgraph probe the offline path uses, ``ctx.policy`` may not exist yet at dry-run
        time for a freshly-constructed module)."""
        meta: dict[str, tuple[int, ...]] = {}
        if "obs_state_dim" in dims:
            state_shape = (int(dims["obs_state_dim"]),)
            for name in ("obs_state", "next_obs_state"):
                if name in fields:
                    meta[name] = state_shape
        if "obs_pixels_shape" in dims:
            pixels_shape = tuple(int(d) for d in dims["obs_pixels_shape"])
            for name in ("obs_pixels", "next_obs_pixels"):
                if name in fields:
                    meta[name] = pixels_shape
        if "action" in fields:
            action_dims = dims.get("multi_action_dims")
            meta["action"] = tuple(int(d) for d in action_dims) if action_dims else ()
        scalar: tuple[int, ...] = ()
        for name in (
            "reward", "terminated", "truncated", "log_prob", "value", "advantage", "return_",
            "episode_start", "mask", "weight", "index", "n_step_discount", "episode_id",
        ):
            if name in fields:
                meta[name] = scalar
        return meta


class _RunnerIterable(IterableDataset):
    """Thin bridge from a configured ``Runner`` to Lightning's training loop."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def __iter__(self) -> Iterator[Any]:
        try:
            yield from self._runner.iter_minibatches()
        finally:
            self._runner.close()


class _StreamEvalDataset(IterableDataset):
    """Capture-only eval iterable: yields per-step collated batches carrying ``episode_id``."""

    def __init__(self, *, regime: InteractiveRegime, module: Any, episodes: int, deterministic: bool, seed: int) -> None:
        self._regime = regime
        self._module = module
        self._episodes = int(episodes)
        self._deterministic = bool(deterministic)
        self._seed = int(seed)

    def __iter__(self) -> Iterator[Any]:
        dm = self._regime._dm
        for record in self._regime.eval_steps(
            self._module, episodes=self._episodes, deterministic=self._deterministic, seed=self._seed
        ):
            yield dm._collate_fn([_eval_seed_item(record)])


def _eval_seed_item(record: dict[str, Any]) -> dict[str, Any]:
    """Unbatched per-sample fields — ``dm._collate_fn`` (called with a length-1 list) supplies
    the sole batch axis via ``Stack``; pre-adding one here would double-batch every field."""
    seed_item: dict[str, Any] = {f"item.obs_{k}": v for k, v in record["obs"].items()}
    seed_item["item.action"] = _as_tensor(record["action"])
    seed_item["item.reward"] = _as_tensor(record["reward"])
    seed_item["item.terminated"] = _as_tensor(record["terminated"])
    seed_item["item.truncated"] = _as_tensor(record["truncated"])
    seed_item["item.mask"] = torch.tensor(record["mask"], dtype=torch.float32)
    seed_item["item.episode_id"] = torch.tensor(record["episode_id"], dtype=torch.long)
    return seed_item


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _policy_device(module: Any) -> torch.device:
    dev = getattr(module, "device", None)
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, str):
        return torch.device(dev)
    return torch.device("cpu")


def _ensure_episode_log(module: Any) -> list[tuple[float, int]] | None:
    if module is None:
        return None
    log = getattr(module, "_rl_episode_log", None)
    if log is None:
        log = []
        module._rl_episode_log = log
    return log
