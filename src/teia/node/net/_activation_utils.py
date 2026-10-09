"""
Shared Activation Decode Utilities.

Source: common knowledge

Description:
  Provides decode math shared by 2+ activation kernels (single-label, multi-label, regression, policy).
  Each function is a pure ``(activated, ctx) -> list[dict]`` callable so an activation's ``kernel``
  staticmethod can delegate to it and have the call collected verbatim into an export bundle.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from teia.core.export.kernels import to_numpy


def decode_regression(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Per-sample ``{"value": [T]}`` rows from activated logits."""
    del ctx
    values = to_numpy(activated["value"])
    return [{"value": row} for row in values.reshape(values.shape[0] if values.ndim else 1, -1)]


def decode_policy(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Per-sample ``{"action": [A]}`` rows from deterministic actions."""
    del ctx
    action = to_numpy(activated["action"])
    return [{"action": row} for row in action.reshape(action.shape[0] if action.ndim else 1, -1)]


def decode_multilabel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Multi-label decode: ``scores`` and multi-hot ``labels`` thresholded per class by ``ctx.thresholds``."""
    probs = to_numpy(activated["probs"])
    probs = probs[None, :] if probs.ndim == 1 else probs
    resolved = np.asarray(resolve_thresholds(ctx.get("thresholds"), probs.shape[-1], float(ctx.get("fallback_threshold", 0.5))))
    return [{"scores": row, "labels": (row >= resolved).astype(np.int64)} for row in probs]


def decode_singlelabel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Single-label decode: ``scores`` and the ``label`` of ``argmax(ctx.weights * scores)`` (plain argmax without weights)."""
    probs = to_numpy(activated["probs"])
    probs = probs[None, :] if probs.ndim == 1 else probs
    weights = np.asarray(resolve_thresholds(ctx.get("weights"), probs.shape[-1], 1.0))
    return [{"scores": row, "label": int((row * weights).argmax())} for row in probs]


def resolve_thresholds(thresholds: Any, count: int, fallback: float) -> list[float]:
    """Broadcast a scalar/list/``None`` threshold spec into one value per class."""
    if thresholds is None:
        return [float(fallback) for _ in range(count)]
    if isinstance(thresholds, (int, float)):
        return [float(thresholds) for _ in range(count)]
    values = [float(item) for item in thresholds]
    if len(values) >= count:
        return values[:count]
    return values + [float(fallback) for _ in range(count - len(values))]


class SquareFrameParams:
    """Activation mixin: publish the square model frame ``image_hw`` into the decode ctx via ``params()``.

    Kernels emit fractional coordinates of the frame they decode into; without restore values
    (capture) that frame is the model input, whose size only the datamodule knows."""

    image_size: int | None = None

    def configure_from_datamodule(self, dm: Any) -> None:
        self.image_size = int(dm.image_size)

    def params(self) -> dict[str, Any]:
        if self.image_size is None:
            raise RuntimeError(f"{type(self).__name__}: image_size unset; configure_from_datamodule must run first.")
        return {"image_hw": (self.image_size, self.image_size)}
