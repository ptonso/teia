"""
Linear Loss-Weight Warmup Callback.

Source:
  - title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    url: "https://arxiv.org/abs/1706.02677"
    year: 2018

Description:
  Gradually increases a selected loss node weight during the first training epochs. Warmup reduces early optimization instability by starting from a smaller effective objective coefficient and moving to the configured weight.

Adaptations:
  - Applies warmup to Teia loss-node weights instead of optimizer learning rates.
"""

from __future__ import annotations

from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from teia.base.keys import sanitize_key


def _normalize_node_alias(value: str) -> str:
    return sanitize_key(str(value).replace("/", "_").replace("-", "_"))


class LossWeightWarmupCallback(Callback):
    """Temporarily downscale a configured loss node's final ``node.weight``."""

    def __init__(
        self,
        loss_attr: str,
        warmup_epochs: int = 20,
        curve: str = "linear",
        start_scale: float = 0.0,
        power: float = 1.0,
    ) -> None:
        super().__init__()
        if not str(loss_attr):
            raise ValueError("LossWeightWarmupCallback requires a non-empty loss_attr.")
        self.loss_attr = _normalize_node_alias(loss_attr)
        self.warmup_epochs = int(warmup_epochs)
        self.curve = str(curve)
        self.start_scale = float(start_scale)
        self.power = float(power)
        if self.warmup_epochs < 0:
            raise ValueError("LossWeightWarmupCallback warmup_epochs must be >= 0.")
        if not 0.0 <= self.start_scale <= 1.0:
            raise ValueError("LossWeightWarmupCallback start_scale must be in [0, 1].")
        if self.curve not in {"linear", "power"}:
            raise ValueError("LossWeightWarmupCallback curve must be 'linear' or 'power'.")
        if self.power <= 0.0:
            raise ValueError("LossWeightWarmupCallback power must be > 0.")
        self._base_weights: dict[str, float] | None = None
        self._term_keys: list[str] | None = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._ensure_terms(pl_module)
        epoch = int(getattr(trainer, "current_epoch", getattr(pl_module, "current_epoch", 0)) or 0)
        self._apply(pl_module, self._scale(epoch))

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        if self._base_weights is not None:
            self._apply(pl_module, 1.0)

    def _ensure_terms(self, pl_module: LightningModule) -> None:
        if self._base_weights is not None:
            return
        rec = self._find_loss_record(pl_module)
        weights = getattr(pl_module, "_loss_term_weight", None)
        if not isinstance(weights, dict):
            raise AttributeError("LossWeightWarmupCallback requires TeiaNetModule._loss_term_weight.")
        terms = list(getattr(rec, "out_key", None) or [])
        if not terms:
            raise ValueError(f"LossWeightWarmupCallback: loss node '{self.loss_attr}' has no loss terms.")
        missing = [key for key in terms if key not in weights]
        if missing:
            raise ValueError(
                f"LossWeightWarmupCallback: loss node '{self.loss_attr}' has unrouted loss terms {missing}."
            )
        self._term_keys = terms
        self._base_weights = {key: float(weights[key]) for key in terms}

    def _find_loss_record(self, pl_module: LightningModule) -> Any:
        for rec in getattr(pl_module, "_loss_records", None) or []:
            if _normalize_node_alias(rec.name) == self.loss_attr:
                return rec
        non_loss_records = [
            *(getattr(pl_module, "_pipeline_records", None) or []),
            *(getattr(pl_module, "_activation_records", None) or []),
        ]
        for rec in non_loss_records:
            if _normalize_node_alias(rec.name) == self.loss_attr:
                raise TypeError(
                    f"LossWeightWarmupCallback: module alias '{self.loss_attr}' is not a loss node."
                )
        raise AttributeError(f"LossWeightWarmupCallback: module has no loss node alias '{self.loss_attr}'.")

    def _scale(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return 1.0
        progress = min(1.0, max(0.0, epoch / self.warmup_epochs))
        warmed = progress if self.curve == "linear" else progress**self.power
        return self.start_scale + (1.0 - self.start_scale) * warmed

    def _apply(self, pl_module: LightningModule, scale: float) -> None:
        assert self._base_weights is not None and self._term_keys is not None
        weights = pl_module._loss_term_weight
        for key in self._term_keys:
            weights[key] = self._base_weights[key] * scale


__all__ = ["LossWeightWarmupCallback"]
