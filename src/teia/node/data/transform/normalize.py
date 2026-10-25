"""``Normalize`` — ``(x * scale - mean) / std`` over a configured ``axis``, with fixed, fitted or running statistics.

One pure ``kernel`` serves training, inference and export. ``stats``: ``fixed`` (literal ``mean``/``std``),
``fit`` (fitted once on the train split at ``plan``) or ``running`` (updated on every call while ``training``,
frozen otherwise). Statistics are per index of ``axis`` and reduce over every other dimension. ``pixel_norm``
(``imagenet`` | ``unit`` | ``signed``) is a preset for image batches (``axis: 1``, ``scale: 1/255``).

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from teia.base.data import Transform

_PRESETS: dict[str, dict[str, Any]] = {
    "imagenet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "unit": {"mean": None, "std": None},
    "signed": {"mean": 0.5, "std": 0.5},
}


def normalize_kernel(value: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    x = value.to(torch.float32)
    if params["scale"] is not None:
        x = x * params["scale"]
    if params["mean"] is not None:
        shape = [1] * x.ndim
        shape[params["axis"]] = -1
        mean = torch.as_tensor(params["mean"], device=x.device, dtype=x.dtype).reshape(shape)
        std = torch.as_tensor(params["std"], device=x.device, dtype=x.dtype).reshape(shape)
        x = (x - mean) / std
    if params["clip"] is not None:
        x = x.clamp(-params["clip"], params["clip"])
    return x, {}


class _Moments:
    """Welford-style running mean and variance per index of ``axis`` (parallel form)."""

    def __init__(self) -> None:
        self.count = 0.0
        self.mean: torch.Tensor | None = None
        self.var: torch.Tensor | None = None

    def update(self, x: torch.Tensor, axis: int) -> None:
        x = x.detach().to(torch.float64).movedim(axis, -1)
        x = x.reshape(-1, x.shape[-1])
        n = float(x.shape[0])
        b_mean, b_var = x.mean(0), x.var(0, unbiased=False)
        if self.mean is None:
            self.count, self.mean, self.var = n, b_mean, b_var
            return
        mean, var = self.mean.to(x.device), self.var.to(x.device)
        total = self.count + n
        delta = b_mean - mean
        m2 = var * self.count + b_var * n + delta.pow(2) * (self.count * n / total)
        self.count, self.mean, self.var = total, mean + delta * (n / total), m2 / total

    def state(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load(self, state: dict[str, Any]) -> None:
        self.count, self.mean, self.var = float(state["count"]), state["mean"], state["var"]


class Normalize(Transform):
    """Normalize ``value`` per index of ``axis``. ``mean``/``std`` (scalar or one per index) are literals for
    ``stats: fixed``; ``fit`` and ``running`` derive them from data. ``clip`` bounds the result to ``[-clip, clip]``."""

    def __init__(
        self,
        *,
        stats: str = "fixed",
        axis: int = -1,
        scale: float | None = None,
        mean: Any = None,
        std: Any = None,
        clip: float | None = None,
        eps: float = 1e-8,
        pixel_norm: str | None = None,
    ) -> None:
        if stats not in ("fixed", "fit", "running"):
            raise ValueError(f"Normalize stats must be fixed, fit or running; got {stats!r}")
        if pixel_norm is not None:
            if pixel_norm not in _PRESETS:
                raise ValueError(f"unknown pixel_norm {pixel_norm!r}; expected one of {sorted(_PRESETS)}")
            axis, scale = 1, 1.0 / 255.0
            mean, std = _PRESETS[pixel_norm]["mean"], _PRESETS[pixel_norm]["std"]
        if (mean is None) != (std is None):
            raise ValueError("Normalize needs mean and std together")
        if stats != "fixed" and mean is not None:
            raise ValueError(f"Normalize stats={stats!r} derives mean/std from data; do not set them")
        self.stats, self.axis, self.scale, self.clip, self.eps = stats, int(axis), scale, clip, eps
        self._mean, self._std = mean, std
        self.stateful = stats != "fixed"
        self.training = True
        self._moments = _Moments()

    def _scaled(self, value: torch.Tensor) -> torch.Tensor:
        x = value.to(torch.float32)
        return x if self.scale is None else x * self.scale

    def fit(self, train_stream: Iterable[Any]) -> None:
        if self.stats != "fit":
            return
        for value in train_stream:
            self._moments.update(self._scaled(torch.as_tensor(value)), self.axis)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if self.stats == "running" and self.training:
            self._moments.update(self._scaled(value), self.axis)
        return normalize_kernel(value, self.params())[0]

    def state(self) -> Any:
        return self._moments.state() if self.stateful else None

    def load_state(self, state: Any) -> None:
        if state is not None:
            self._moments.load(state)

    def params(self) -> dict[str, Any]:
        mean, std = self._mean, self._std
        if self.stateful and self._moments.mean is not None:
            mean = self._moments.mean.tolist()
            std = self._moments.var.clamp_min(self.eps).sqrt().tolist()
        return {"axis": self.axis, "scale": self.scale, "mean": mean, "std": std, "clip": self.clip}

    kernel = staticmethod(normalize_kernel)
