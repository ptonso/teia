from typing import Any, Iterable

from teia.base.data import Reader
from teia.base.interactive.env import BaseRlIO

class EnvReader(Reader):
    """Component-based graph node for interactive interactive environments."""

    def __init__(self, io: BaseRlIO, num_envs: int, seed: int = 0, total_env_steps: int = 0, eval_episodes: int = 10, **kwargs: Any) -> None:
        self.io = io
        self.num_envs = num_envs
        self.seed = seed
        self.total_env_steps = total_env_steps
        self.eval_episodes = eval_episodes
        self.env_spec = self.io.probe_spec()
        self._env = None

    def runtime_dims(self) -> dict[str, Any]:
        dims = {}
        spec = self.env_spec
        if "state" in spec.observation_shapes:
            dims["obs_state_dim"] = spec.observation_shapes["state"][0]
        if "pixels" in spec.observation_shapes:
            dims["obs_pixels_shape"] = spec.observation_shapes["pixels"]
        
        dims["multi_action_dims"] = spec.action_shape
        dims["action_dim"] = spec.action_shape[0] if spec.action_shape else 0
        if spec.action_space == "discrete":
            dims["num_actions"] = spec.action_shape[0] if spec.action_shape else 0
        return dims

    def sampling_generator(self):
        return None

    def setup_stream(self) -> None:
        """Called by the interactive Runner before the training loop starts."""
        self._env = self.io.build_train_env(self.seed, self.num_envs)

    def reset(self) -> Any:
        if self._env is None:
            raise RuntimeError("Called reset() before setup_stream()")
        return self.io.reset(self._env)

    def step(self, action: Any) -> Any:
        if self._env is None:
            raise RuntimeError("Called step() before setup_stream()")
        return self.io.step(self._env, action)

    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        """Reader ABC contract, but interactive interactive uses the streaming Runner interface."""
        raise NotImplementedError("EnvReader is driven iteratively by the Runner via step(), not iter_split().")
