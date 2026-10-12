"""
FrechetDistance: FID-style unpaired distributional distance.

Source:
  - title: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    url: "https://arxiv.org/abs/1706.08500"
    year: 2017

Description:
  Unpaired metrics compare aggregate prediction and target distributions in a feature space
  rather than per-sample pairs, since no correspondence exists (a generated image has no matching
  ground-truth image). Predictions and references are each embedded via a nested
  ``feature_extractor`` (see ``features.py``), fit to a Gaussian, and compared by the Fréchet
  distance between the two Gaussians. The real side is either embedded from ``real_paths`` or
  read from ``reference``, a ``.npz`` of precomputed statistics written by ``save_reference``
  (``mu``, ``sigma``, ``n``, ``extractor``), so a run need not re-embed a large real set.

Streaming-first: ``update``/``compute`` are the primary path and ``from_source`` builds on
them, because the predictions this metric is for (video, high-resolution generation) are the
case that cannot be materialized for a whole split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from teia.base.eval import Metric
from teia.node.eval.metric.unpaired._embedding import EmbeddingAccumulator
from teia.node.eval.metric.unpaired._shared import frechet_distance

__all__ = ["FrechetDistance", "reference_statistics", "save_reference"]


def reference_statistics(paths: list[str], extractor: Any, *, device: str = "cpu", batch_size: int = 256) -> dict[str, Any]:
    """Mean and covariance of ``extractor`` features over the images at ``paths``."""
    accumulator = EmbeddingAccumulator(extractor, device=device, batch_size=batch_size)
    accumulator.update_paths(paths)
    mu, sigma = accumulator.mean_and_cov()
    return {"mu": mu, "sigma": sigma, "n": accumulator.n, "extractor": type(extractor).identity}


def save_reference(path: str | Path, stats: dict[str, Any]) -> Path:
    """Write ``reference_statistics`` output as the ``.npz`` that ``FrechetDistance(reference=...)`` reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=stats["mu"], sigma=stats["sigma"], n=stats["n"], extractor=stats["extractor"])
    return path


class FrechetDistance(Metric):
    """Constructor kwargs: ``feature_extractor`` (nested ``_target_``, a plain ``nn.Module``;
    see ``features.py``), ``device`` (default ``"cpu"``; no trainer is available to supply one),
    ``reference`` (path to a ``save_reference`` file standing in for the real side) and
    ``batch_size`` (images per extractor call)."""

    def __init__(self, feature_extractor: Any, device: str = "cpu", reference: str | None = None, batch_size: int = 256) -> None:
        self._pred = EmbeddingAccumulator(feature_extractor, device=device, batch_size=batch_size)
        self._real = EmbeddingAccumulator(feature_extractor, device=device, batch_size=batch_size)
        self.reference = None if reference is None else _load_reference(reference, type(feature_extractor).identity)

    def update(self, *, pred: torch.Tensor | None = None, real: torch.Tensor | None = None, **_: Any) -> None:
        if pred is not None:
            self._pred.update_tensors(pred)
        if real is not None:
            if self.reference is not None:
                raise ValueError("FrechetDistance: got real images and a reference; wire one of them.")
            self._real.update_tensors(real)

    def compute(self) -> dict[str, Any]:
        if self._pred.n == 0:
            raise ValueError("FrechetDistance: no predicted images.")
        pred_mean, pred_cov = self._pred.mean_and_cov()
        if self.reference is not None:
            real_mean, real_cov, n_real = self.reference["mu"], self.reference["sigma"], int(self.reference["n"])
        elif self._real.n > 0:
            (real_mean, real_cov), n_real = self._real.mean_and_cov(), self._real.n
        else:
            raise ValueError("FrechetDistance: no real images and no reference.")
        distance = round(frechet_distance(pred_mean, pred_cov, real_mean, real_cov), 6)
        return {
            "n_pred": self._pred.n,
            "n_real": n_real,
            "frechet_distance": distance,
            # Pre-shaped for Table's `rows: list[dict]` kwarg.
            "rows": [{"metric": "frechet_distance", "value": distance, "n_pred": self._pred.n, "n_real": n_real}],
        }

    def reset(self) -> None:
        self._pred.reset()
        self._real.reset()

    def from_source(self, source: Any, *, pred_paths: list[str] | None = None, real_paths: list[str] | None = None, **_: Any) -> dict[str, Any]:
        del source
        if bool(real_paths) == (self.reference is not None):
            raise ValueError("FrechetDistance: wire exactly one of `real_paths` and `reference`.")
        self.reset()
        self._pred.update_paths(list(pred_paths or []))
        if real_paths:
            self._real.update_paths(list(real_paths))
        return self.compute()


def _load_reference(path: str, identity: str) -> dict[str, Any]:
    with np.load(path) as data:
        stats = {key: data[key] for key in ("mu", "sigma", "n", "extractor")}
    if str(stats["extractor"]) != identity:
        raise ValueError(f"FrechetDistance: reference {path} was computed with {str(stats['extractor'])!r}, the metric uses {identity!r}.")
    return stats
