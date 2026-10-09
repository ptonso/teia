"""Base environment IO contract.

IO adapters own environment creation, vectorization, reset/step normalization, render
capture, and conversion into the canonical transition contract. IO owns the env-step
result (``EnvStepResult``); the collection loop owns the ``TransitionRecord``.

**Rendering is native to the IO.** Each backend renders in its own optimal way; teia never
reconstructs a scene from observations. ``render_mode`` selects how:

- ``"none"`` — no rendering.
- ``"human"`` — the backend opens/drives its **own** interactive window (for example a native viewer). teia mounts no display code.
- ``"rgb_array"`` — the backend produces frames; ``render(env)`` returns them and a generic
  teia viewer (or the report) displays/captures them.

``render_mode`` governs only human-facing visualization. **Pixels-as-observation**
(``obs_pixels`` for a pixel policy) is a separate concern driven by ``pixels_shape``
or ``obs_mode``, independent of ``render_mode``.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from torch import Tensor

from teia.base.interactive.batch import EnvSpec, EnvStepResult


class BaseRlIO(ABC):
    """Adapter between ``RlDataModule`` and a concrete environment backend.

    ``probe_spec()`` is the cheap eager probe (build one env, read spaces, close)
    called at ``RlDataModule.__init__``; the heavyweight vectorized envs are built
    later in ``setup("fit")`` via ``build_train_env`` / ``build_eval_env``.

    ``RENDER_MODES`` advertises the render modes a backend supports (subclasses that have a
    native interactive window add ``"human"``); ``_set_render_mode`` validates against it.
    """

    # Backends without a native interactive window support only offscreen frames.
    RENDER_MODES: tuple[str, ...] = ("none", "rgb_array")

    def __init__(self) -> None:
        self._spec: EnvSpec | None = None
        self.render_mode: str = "none"

    def _set_render_mode(self, render_mode: Any) -> str:
        """Normalize/validate ``render_mode`` against this backend's ``RENDER_MODES``."""
        mode = str(render_mode or "none").lower()
        if mode not in type(self).RENDER_MODES:
            raise ValueError(
                f"{type(self).__name__} does not support render_mode={mode!r}; "
                f"supported modes are {type(self).RENDER_MODES}."
            )
        self.render_mode = mode
        return mode

    @property
    def supports_render(self) -> bool:
        """Whether a human-facing render (viewer/report) is available (``render_mode != none``)."""
        return self.render_mode != "none"

    @abstractmethod
    def probe_spec(self) -> EnvSpec:
        """Cheaply derive the ``EnvSpec``: build one env, read spaces, close it."""

    @abstractmethod
    def build_train_env(self, seed: int, num_envs: int) -> Any:
        """Build the vectorized training environment with strict seed splitting."""

    @abstractmethod
    def build_eval_env(self, seed: int, num_envs: int, *, view: bool = False) -> Any:
        """Build the vectorized evaluation environment with strict seed splitting.

        ``view=True`` is the live-view env: the only env that honors the viewer
        ``render_mode`` (``human`` window / ``rgb_array`` frames). Collection, metric-eval,
        and ``probe_spec`` build with ``view=False`` so they never open a viewer window.
        """

    @abstractmethod
    def reset(self, env: Any, *, seed: int | None = None) -> Mapping[str, Tensor]:
        """Reset ``env`` and return the flattened ``obs_*`` mapping."""

    @abstractmethod
    def step(self, env: Any, action: Tensor) -> EnvStepResult:
        """Step ``env`` with ``action`` and return environment-only fields."""

    def render(self, env: Any) -> Tensor | None:
        """Optional render capture for reports; ``None`` when unsupported."""
        return None

    def spec(self) -> EnvSpec:
        """Cached ``EnvSpec`` from ``probe_spec`` / build (probes lazily once)."""
        if self._spec is None:
            self._spec = self.probe_spec()
        return self._spec
