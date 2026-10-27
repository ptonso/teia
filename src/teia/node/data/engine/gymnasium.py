"""Gymnasium vector-env engine (Box / Discrete actions, Box or Dict-of-Box observations).

Signals: ``observation`` (a Dict space gives one ``observation_<key>`` per key in sorted order), ``reward``,
``terminated``, ``truncated`` and ``episode_step``. The world is ``data_root=gym:<env id>``.

The vector env is built with Gymnasium's disabled autoreset: a finished environment keeps its final values until the
runner resets it through ``reset(mask)``, which maps to Gymnasium's ``reset_mask`` option.

Source:
  - title: "Gymnasium: A Standard Interface for Reinforcement Learning Environments"
    url: "https://arxiv.org/abs/2407.17032"
    year: 2024

Description:
  Wraps the Gymnasium vector API into the signal Engine contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from teia.node.data.engine._vector import OUTCOME, STATE, ActionSpace, VectorEngine, dtype_name, to_tensor, world_id


class GymnasiumEngine(VectorEngine):
    """Gymnasium vector env with disabled autoreset; see the module docstring for the signals."""

    def __init__(
        self,
        *,
        vectorization_mode: str = "sync",
        render_mode: str = "none",
        env_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if render_mode not in ("none", "rgb_array"):
            raise ValueError("render_mode is none|rgb_array (a vector env would open one window per environment).")
        self.vectorization_mode, self.render_mode = vectorization_mode, render_mode
        self.env_kwargs = dict(env_kwargs or {})
        self._seed_pending: int | None = None

    # -- construction -------------------------------------------------------------
    def _gym_render_mode(self) -> str | None:
        return "rgb_array" if self.render_mode == "rgb_array" else None

    def _build(self, seed: int) -> Any:
        import gymnasium as gym

        env = gym.make_vec(
            world_id(self.data_root, "gym"),
            num_envs=self.num_envs,
            vectorization_mode=self.vectorization_mode,
            vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.DISABLED},
            render_mode=self._gym_render_mode(),
            **self.env_kwargs,
        )
        self._seed_pending = seed
        space = env.single_observation_space
        self._keys = sorted(space.spaces) if isinstance(space, gym.spaces.Dict) else None
        self._obs_space, self._act_space = space, env.single_action_space
        return env

    # -- spaces -------------------------------------------------------------------
    def _observation_names(self) -> list[str]:
        return ["observation"] if self._keys is None else [f"observation_{k}" for k in self._keys]

    def _leaves(self) -> list[Any]:
        space = self._obs_space
        return [space.spaces[k] for k in self._keys] if self._keys is not None else [space]

    def _catalog(self) -> dict[str, tuple[tuple[int, ...], str, str]]:
        states = {
            name: (tuple(leaf.shape), dtype_name(to_tensor(np.zeros(leaf.shape, leaf.dtype))), STATE)
            for name, leaf in zip(self._observation_names(), self._leaves())
        }
        outcomes = {"reward": ((), "float32", OUTCOME), "terminated": ((), "bool", OUTCOME), "truncated": ((), "bool", OUTCOME)}
        return {**states, **outcomes}

    def _action_space(self) -> ActionSpace:
        import gymnasium as gym

        space = self._act_space
        if isinstance(space, gym.spaces.Box):
            return ActionSpace(tuple(space.shape), "float32", torch.as_tensor(space.low).float(), torch.as_tensor(space.high).float())
        if isinstance(space, gym.spaces.Discrete):
            return ActionSpace((), "int64", num_actions=int(space.n))
        raise TypeError(f"GymnasiumEngine supports Box and Discrete actions; got {type(space).__name__}")

    # -- stepping -----------------------------------------------------------------
    def _observation(self, obs: Any) -> dict[str, Tensor]:
        leaves = [obs] if self._keys is None else [obs[k] for k in self._keys]
        return {name: to_tensor(leaf) for name, leaf in zip(self._observation_names(), leaves)}

    def _extras(self) -> dict[str, Tensor]:
        """Hook for subclasses that add render-based signals."""
        return {}

    def _do_reset(self, mask: Tensor | None) -> dict[str, Tensor]:
        seed, self._seed_pending = self._seed_pending, None
        if mask is None:
            obs, _ = self._sim.reset(seed=seed)
        else:
            obs, _ = self._sim.reset(options={"reset_mask": mask.detach().cpu().numpy().astype(np.bool_)})
        zeros = torch.zeros(self.num_envs)
        return {**self._observation(obs), "reward": zeros, "terminated": zeros.bool(), "truncated": zeros.bool(), **self._extras()}

    def _do_step(self, action: Tensor) -> dict[str, Tensor]:
        obs, reward, terminated, truncated, _ = self._sim.step(action.detach().to("cpu").numpy())
        return {
            **self._observation(obs),
            "reward": torch.as_tensor(np.asarray(reward), dtype=torch.float32),
            "terminated": torch.as_tensor(np.asarray(terminated)),
            "truncated": torch.as_tensor(np.asarray(truncated)),
            **self._extras(),
        }

    def render(self) -> Tensor | None:
        if self.render_mode != "rgb_array" or self._sim is None:
            return None
        return torch.as_tensor(np.asarray(self._sim.render()))
