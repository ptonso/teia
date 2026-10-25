"""Live policy-rollout viewer callback for stream mode.

Runs one deterministic rollout on the open engine on a training-step cadence and shows the frames the engine's
``render()`` returns in a generic OpenCV window, one capture-slice act per step. Self-disables unless
``datamodule.view.enabled`` is set, so the view is a datamodule config characteristic, and for engines without
frames. The rollout uses the same cycle as evaluation (frozen item nodes), and the engine is reset around it.
"""

from __future__ import annotations

from typing import Any, Mapping

from lightning.pytorch.callbacks import Callback

from teia.core.interactive.render import OpenCvFrameViewer


class LiveViewCallback(Callback):
    def __init__(self, *, every_n_steps: int = 2000, max_steps: int = 400, deterministic: bool = True, fps: int = 60) -> None:
        self._every_n_steps = int(every_n_steps)
        self._max_steps = int(max_steps)
        self._deterministic = bool(deterministic)
        self._fps = int(fps)
        self._viewer: OpenCvFrameViewer | None = None
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
        self._rollout(datamodule, pl_module, cfg)

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

    def _rollout(self, datamodule: Any, module: Any, cfg: Mapping[str, Any]) -> None:
        engine = datamodule._engine_node
        limit = int(cfg.get("max_steps", self._max_steps)) * engine.spec().num_envs
        deterministic = bool(cfg.get("deterministic", self._deterministic))
        for _ in datamodule._regime.play(module, until=lambda cycle: cycle.env_steps >= limit, deterministic=deterministic):
            frame = engine.render()
            if frame is None:
                return
            if self._viewer is None:
                self._viewer = OpenCvFrameViewer(fps=int(cfg.get("fps", self._fps)))
                self._viewer.open()
            self._viewer.show(frame)

    def _teardown(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


__all__ = ["LiveViewCallback"]
