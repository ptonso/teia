"""Generic engine training callbacks callable from Hydra ``_target_``."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import RichModelSummary, WeightAveraging
from torch import nn


def ramped_ema_avg_fn(
    decay: float = 0.9999,
    tau: float = 2000.0,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """EMA averaging fn with exponential warm-up ramp.

    Per-step decay is ``decay * (1 - exp(-step / tau))``. Consumed by
    :class:`TeiaWeightAveraging` (a ``lightning.pytorch.callbacks.WeightAveraging``).
    """
    def _fn(averaged_param: torch.Tensor, current_param: torch.Tensor, num_averaged: torch.Tensor) -> torch.Tensor:
        step = float(num_averaged.item()) + 1.0
        d = decay * (1.0 - math.exp(-step / tau))
        return d * averaged_param + (1.0 - d) * current_param

    return _fn


class TeiaWeightAveraging(WeightAveraging):
    """Ramped-EMA weight averaging that updates **only trainable parameters**.

    A frozen backbone (``encoder.freeze=true``) otherwise pays a full O(params) copy every step
    to maintain a numerically constant average. We capture the trainable mask at fit start and
    apply the ramped EMA only to ``requires_grad`` params; buffers (BN stats) keep averaging.
    """

    def __init__(self, *, decay: float = 0.9999, tau: float = 2000.0, use_buffers: bool = True, device: Any = None) -> None:
        self._ema_fn = ramped_ema_avg_fn(decay=decay, tau=tau)
        self._use_buffers = bool(use_buffers)
        self._trainable_mask: list[bool] | None = None
        super().__init__(device=device, use_buffers=use_buffers, multi_avg_fn=self._multi_avg_fn)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            mask = [p.requires_grad for p in pl_module.parameters()]
            if self._use_buffers:
                mask += [True] * sum(1 for _ in pl_module.buffers())
            self._trainable_mask = mask
        super().setup(trainer, pl_module, stage)

    def _multi_avg_fn(self, averaged_params: Any, current_params: Any, num_averaged: Any) -> None:
        mask = self._trainable_mask
        if mask is None or len(mask) != len(averaged_params):
            mask = [True] * len(averaged_params)
        for avg_p, cur_p, trainable in zip(averaged_params, current_params, mask):
            if trainable:
                avg_p.copy_(self._ema_fn(avg_p, cur_p, num_averaged))


def _safe_parse_model_summary_shape(batch: Any) -> Any:
    """Lightning model-summary shape parser that treats metadata as unknown size.

    Lightning's parser assumes any object with a ``shape`` attribute has an iterable
    shape. Detection-family raw outputs can contain optional fields, bound methods,
    modules, or metadata objects whose ``shape`` is ``None``. Those are valid model
    outputs, but they are not useful shape rows, so the summary should show ``?``.
    """
    from lightning.pytorch.utilities.model_summary import model_summary as lightning_model_summary

    unknown = getattr(lightning_model_summary, "UNKNOWN_SIZE", "?")
    if batch is None or isinstance(batch, (str, bytes, bytearray, nn.Module)) or callable(batch):
        return unknown

    try:
        shape = getattr(batch, "shape", None)
    except Exception:
        return unknown
    if shape is not None:
        try:
            return list(shape)
        except Exception:
            return unknown

    if isinstance(batch, Mapping):
        return {str(key): _safe_parse_model_summary_shape(value) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [_safe_parse_model_summary_shape(item) for item in batch]

    return unknown


class GradSafeRichModelSummary(RichModelSummary):
    """RichModelSummary whose FLOP/shape example forward never touches grad state.

    Lightning runs the summary's example forward under ``torch.no_grad()`` with a
    ``FlopCounterMode`` that installs ``register_multi_grad_hook`` on every
    grad-requiring input. A trainable leaf parameter reached through a view op
    (e.g. SigLIP attn-pool's ``latent.expand(...)``) is then ``requires_grad=True``
    with ``grad_fn=None`` under no_grad, which the hook rejects
    (``AssertionError: Expected gradient function to be set``).

    The summary is measurement-only, so no parameter needs grad: we disable grad on
    every parameter for the duration of the summary and restore it afterwards. This
    is model-agnostic — it sidesteps the whole class of view-of-leaf params without
    knowing anything about the architecture.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from lightning.pytorch.utilities.model_summary import model_summary as lightning_model_summary

        saved = [(p, p.requires_grad) for p in pl_module.parameters()]
        original_parse_batch_shape = lightning_model_summary.parse_batch_shape
        for p, _ in saved:
            p.requires_grad_(False)
        try:
            lightning_model_summary.parse_batch_shape = _safe_parse_model_summary_shape
            super().on_fit_start(trainer, pl_module)
        finally:
            lightning_model_summary.parse_batch_shape = original_parse_batch_shape
            for p, state in saved:
                p.requires_grad_(state)
