"""Live policy-rollout viewer callback for the interactive regime.

Runs one deterministic policy rollout on a training-step cadence and displays it via the
IO-native render path (config-driven; rendering is native to the IO). Self-disables unless
``datamodule.view.enabled`` is set, so the view is a datamodule config characteristic. Three
display paths, chosen from ``datamodule.view`` + ``io.render_mode``: a custom ``LiveRenderer``
(obs-interpreting drawing, for state-only envs); ``rgb_array`` frames shown in a generic OpenCV
window; or human mode where the backend draws its own window. Reuses the datamodule's data graph
and PolicyAdapter so the rollout path matches collection exactly (one forward per step, no_grad;
frozen normalization).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from lightning.pytorch.callbacks import Callback

from teia.core.interactive.render import LiveRenderer, LiveStepRecord, OpenCvFrameViewer


class LiveViewCallback(Callback):
    """Roll out the live policy on a cadence and display it via the IO-native render path."""

    def __init__(
        self,
        *,
        every_n_steps: int = 2000,
        max_steps: int = 400,
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        self._every_n_steps = int(every_n_steps)
        self._max_steps = int(max_steps)
        self._deterministic = bool(deterministic)
        self._seed = int(seed)
        self._custom: LiveRenderer | None = None
        self._frame_viewer: OpenCvFrameViewer | None = None
        self._env: Any = None
        self._opened = False
        self._last_fired = -1

    def on_train_batch_end(self, trainer: Any, pl_module: Any, *args: Any, **kwargs: Any) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        cfg = self._view_cfg(datamodule)
        if not cfg or not bool(cfg.get("enabled", False)):
            return
        every = int(cfg.get("every_n_steps", self._every_n_steps))
        step = int(trainer.global_step)
        if every <= 0 or step == self._last_fired or step % every != 0:
            return
        self._last_fired = step
        self._rollout(datamodule, pl_module, cfg, seed=self._seed + step)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self._teardown()

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
        self._teardown()

    @staticmethod
    def _view_cfg(datamodule: Any) -> Mapping[str, Any]:
        view = getattr(datamodule, "view", None)
        if view is None:
            return {}
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(view):
                return dict(OmegaConf.to_container(view, resolve=True))  # type: ignore[arg-type]
        except Exception:
            pass
        return dict(view) if isinstance(view, Mapping) else {}

    def _build_renderer(self, datamodule: Any, cfg: Mapping[str, Any]) -> None:
        """Instantiate the display sink once, from ``view`` + the IO's ``render_mode``."""
        if self._opened:
            return
        renderer_cfg = cfg.get("renderer")
        render_mode = str(getattr(self._io(datamodule), "render_mode", "none"))
        if renderer_cfg is not None:
            self._custom = self._instantiate_renderer(renderer_cfg)
            self._custom.open(datamodule.env_spec)
        elif render_mode == "rgb_array":
            self._frame_viewer = OpenCvFrameViewer(fps=int(cfg.get("fps", 60)))
            self._frame_viewer.open()
        self._opened = True

    @staticmethod
    def _instantiate_renderer(renderer: Any) -> LiveRenderer:
        if isinstance(renderer, LiveRenderer):
            return renderer
        from teia.core.instantiate import instantiate

        obj = instantiate(renderer)
        if not isinstance(obj, LiveRenderer):
            raise TypeError(
                f"datamodule.view.renderer must instantiate a LiveRenderer, got {type(obj).__name__}."
            )
        return obj

    @staticmethod
    def _io(datamodule: Any) -> Any:
        return datamodule._stream_reader_node().io

    def _rollout(self, datamodule: Any, module: Any, cfg: Mapping[str, Any], *, seed: int) -> None:
        from teia.core.datamodule.interactive import FeedforwardPolicyAdapter

        io = self._io(datamodule)
        self._build_renderer(datamodule, cfg)

        deterministic = bool(cfg.get("deterministic", self._deterministic))
        max_steps = int(cfg.get("max_steps", self._max_steps))
        device = getattr(module, "device", torch.device("cpu"))
        adapter = FeedforwardPolicyAdapter(module)
        gen = torch.Generator()
        gen.manual_seed(seed)
        # Build the viewer env once and reuse it across cadences so the backend's native
        # window (render_mode='human') is a single, coherent window — not a new one per
        # cadence. Each cadence resets it to a fresh episode (closed in _teardown).
        if self._env is None:
            self._env = io.build_eval_env(seed, 1, view=True)
        env = self._env
        raw_obs = io.reset(env, seed=seed)
        for i in range(max_steps):
            seed_ws = {f"item.obs_{k}": v.to(device) for k, v in raw_obs.items()}
            seed_ws["ctx.policy"] = module
            workspace = datamodule._item_workspace(seed_ws, augment=False)
            
            class _LiveBatch:
                def __getattr__(self, key: str) -> Any:
                    return None

            obs_batch = _LiveBatch()
            for k, v in workspace.items():
                if k.startswith("item.obs_"):
                    setattr(obs_batch, k[5:], v)
            
            sample = adapter.act(obs_batch, deterministic=deterministic, generator=gen)
            env_action = sample.env_action.detach().to("cpu")
            step_result = io.step(env, env_action)
            self._display(io, env, i, raw_obs, env_action, step_result)
            raw_obs = step_result.next_obs

    def _display(self, io: Any, env: Any, i: int, raw_obs: Any, env_action: Any, step_result: Any) -> None:
        if self._custom is not None:
            self._custom.show(
                LiveStepRecord(
                    step=i,
                    obs=raw_obs,
                    action=env_action,
                    reward=step_result.reward,
                    terminated=step_result.terminated,
                    truncated=step_result.truncated,
                )
            )
        elif self._frame_viewer is not None:
            frame = io.render(env)
            if frame is not None:
                self._frame_viewer.show(frame)
        else:
            # 'human' mode handles its own window but still requires a render() tick
            io.render(env)

    def _teardown(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None
        if not self._opened:
            return
        if self._custom is not None:
            self._custom.close()
            self._custom = None
        if self._frame_viewer is not None:
            self._frame_viewer.close()
            self._frame_viewer = None
        self._opened = False


__all__ = ["LiveViewCallback"]
