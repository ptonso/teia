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
  distance between the two Gaussians.

Streaming-first: ``update``/``compute`` are the primary path and ``from_source`` builds on
them, because the predictions this metric is for (video, high-resolution generation) are the
case that cannot be materialized for a whole split.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.eval import Metric
from teia.node.eval.metric.unpaired._embedding import EmbeddingAccumulator
from teia.node.eval.metric.unpaired._shared import frechet_distance

__all__ = ["FrechetDistance"]


class FrechetDistance(Metric):
    """Constructor kwargs: ``feature_extractor`` (nested ``_target_``, a plain ``nn.Module``;
    see ``features.py`` for why it must not be a ``TeiaNode``) and ``device`` (default
    ``"cpu"``; no trainer is available to supply one)."""

    def __init__(self, feature_extractor: Any, device: str = "cpu") -> None:
        self._extractor_config = feature_extractor
        self.device = device
        self._pred = EmbeddingAccumulator(feature_extractor, device=device)
        self._real = EmbeddingAccumulator(feature_extractor, device=device)

    def update(self, *, pred: torch.Tensor | None = None, real: torch.Tensor | None = None, **_: Any) -> None:
        if pred is not None:
            self._pred.update_tensors(pred)
        if real is not None:
            self._real.update_tensors(real)

    def compute(self) -> dict[str, Any]:
        if self._pred.n == 0 or self._real.n == 0:
            return {"n_pred": self._pred.n, "n_real": self._real.n}
        pred_mean, pred_cov = self._pred.mean_and_cov()
        real_mean, real_cov = self._real.mean_and_cov()
        distance = round(frechet_distance(pred_mean, pred_cov, real_mean, real_cov), 6)
        return {
            "n_pred": self._pred.n,
            "n_real": self._real.n,
            "frechet_distance": distance,
            # Pre-shaped for Table's `rows: list[dict]` kwarg.
            "rows": [{"metric": "frechet_distance", "value": distance, "n_pred": self._pred.n, "n_real": self._real.n}],
        }

    def reset(self) -> None:
        self._pred.reset()
        self._real.reset()

    def from_source(self, source: Any, *, pred_paths: list[str] | None = None, real_paths: list[str] | None = None, **_: Any) -> dict[str, Any]:
        del source
        self.reset()
        if pred_paths:
            self._pred.update_paths(list(pred_paths))
        if real_paths:
            self._real.update_paths(list(real_paths))
        return self.compute()
