"""
``gen_diagnostics`` metric: per-epoch reconstruction / KL diagnostics of a latent-variable model.

Source: common knowledge

Description:
  Reads the run's ``metrics.csv`` (``loss.recon``, ``loss.kld`` per stage) and its composed config
  (decoder ``pred.reconstructed`` shape or ``datamodule.image_size`` for the reconstruction element
  count; neck ``feat.z`` shape for the latent dim; ``GaussianKLDLoss.weight`` for β) and normalizes
  each series to a per-pixel / per-latent-dim scale a ``MultiCurveBundle`` view renders. Offline only.
"""

from __future__ import annotations

import math
from typing import Any

from teia.base.eval import Metric
from teia.node.eval.metric.run.training_curves import _numeric_columns

__all__ = ["GenDiagnostics"]


class GenDiagnostics(Metric):
    """Constructor kwargs: ``logger`` (CSV logger name), ``step_key`` (x-axis column, default
    ``"epoch"``), ``default_beta`` (β fallback when no KLD-loss node declares a weight)."""

    def __init__(self, logger: str = "teia", step_key: str = "epoch", default_beta: float = 2.0) -> None:
        self.logger = logger
        self.step_key = step_key
        self.default_beta = float(default_beta)

    def from_source(self, source: Any, **_: Any) -> dict[str, Any]:
        curves = source.curves() or {}
        rows = curves.get(self.logger) or (max(curves.values(), key=len) if curves else [])
        if not rows:
            return {"n_rows": 0, "recon_curves": [], "kld_curves": []}
        columns = _numeric_columns(rows)
        config = dict(source.config() or {})
        x_key = self.step_key if self.step_key in columns else ("step" if "step" in columns else None)
        x = columns.get(x_key) if x_key else None

        recon_columns = _stage_columns(columns, "loss.recon")
        kld_columns = _stage_columns(columns, "loss.kld")
        if not recon_columns and not kld_columns:
            return {"n_rows": len(rows), "recon_curves": [], "kld_curves": []}

        beta = _vae_beta(config, self.default_beta)
        recon_curves: list[dict[str, Any]] = []
        kld_curves: list[dict[str, Any]] = []
        recon_elements: int | None = None
        latent_dim: int | None = None
        if recon_columns:
            recon_elements = _recon_element_count(config)
            recon_curves = [_curve(f"{stage}/recon_pixel", x, _divide(values, float(recon_elements))) for stage, values in recon_columns.items()]
        if kld_columns:
            latent_dim = _latent_dim(config)
            for stage, values in kld_columns.items():
                kld_curves.append(_curve(f"{stage}/kld_weighted_per_dim", x, _divide(values, float(latent_dim))))
                kld_curves.append(_curve(f"{stage}/kld_raw_per_dim", x, _divide(values, float(latent_dim) * beta)))

        return {
            "n_rows": len(rows),
            "recon_elements": recon_elements,
            "latent_dim": latent_dim,
            "beta": beta,
            "recon_curves": recon_curves,
            "kld_curves": kld_curves,
        }


def _stage_columns(columns: dict[str, list[float]], metric: str) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for stage in ("train", "val"):
        for key in (f"{stage}/{metric}_epoch", f"{stage}/{metric}", f"{stage}/{metric}_step"):
            if key in columns:
                found[stage] = columns[key]
                break
    return found


def _divide(values: list[float], denominator: float) -> list[float]:
    return [value / denominator for value in values]


def _curve(label: str, x: list[float] | None, values: list[float]) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, float) and math.isnan(value):
            continue
        xs.append(float(x[index]) if x and index < len(x) and not math.isnan(x[index]) else float(index))
        ys.append(float(value))
    return {"label": label, "x": xs, "y": ys}


def _recon_element_count(config: dict[str, Any]) -> int:
    shape = _declared_shape(config, "pred.reconstructed")
    if shape is not None:
        return _product(shape, name="pred.reconstructed")
    datamodule = config.get("datamodule") or {}
    if "image_size" not in datamodule:
        raise ValueError("gen_diagnostics needs the decoder pred.reconstructed shape or datamodule.image_size.")
    image_size = int(datamodule["image_size"])
    return 3 * image_size * image_size


def _latent_dim(config: dict[str, Any]) -> int:
    shape = _declared_shape(config, "feat.z")
    if shape is None:
        raise ValueError("gen_diagnostics needs the latent declared on the neck's feat.z out shape.")
    return _product(shape, name="feat.z")


def _declared_shape(config: dict[str, Any], key: str) -> list[Any] | None:
    for entry in (config.get("netmodule") or {}).values():
        if not isinstance(entry, dict):
            continue
        out_key = entry.get("out") or {}
        if isinstance(out_key, dict) and out_key.get(key) is not None:
            return list(out_key[key])
    return None


def _product(shape: list[Any], *, name: str) -> int:
    if not shape:
        raise ValueError(f"gen_diagnostics needs a non-empty shape for {name}.")
    product = 1
    for dim in shape:
        try:
            product *= int(dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gen_diagnostics needs integer dims for {name}: {shape!r}.") from exc
    return product


def _vae_beta(config: dict[str, Any], default: float) -> float:
    for entry in (config.get("netmodule") or {}).values():
        if isinstance(entry, dict) and str(entry.get("_target_", "")).endswith("GaussianKLDLoss"):
            beta = float(entry.get("weight", default))
            if beta <= 0.0:
                raise ValueError("gen_diagnostics needs loss.kld weight (β) > 0 for raw-KL normalization.")
            return beta
    return default
