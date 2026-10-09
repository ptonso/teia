"""
KernelDistance: MMD-style unpaired distributional distance.

Source:
  - title: "A Kernel Two-Sample Test"
    url: "https://jmlr.org/papers/v13/gretton12a.html"
    year: 2012
  - title: "Demystifying MMD GANs"
    url: "https://arxiv.org/abs/1801.01401"
    year: 2018

Description:
  Same embedding-then-compare shape as :class:`~.frechet_distance.FrechetDistance` (see its module
  docstring for the streaming-first rationale), but compares the two embedding sets via Gretton et
  al.'s unbiased Maximum Mean Discrepancy estimator instead of fitting Gaussians, so it assumes
  nothing about the distribution shape.

Adaptations:
  - KID-style, but not KID. Bińkowski et al. define KID with a degree-3 polynomial kernel; this
    uses an RBF kernel whose bandwidth defaults to the median-pairwise-distance heuristic. Values
    are therefore not comparable with published KID numbers.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.eval import Metric
from teia.node.eval.metric.unpaired._embedding import EmbeddingAccumulator
from teia.node.eval.metric.unpaired._shared import rbf_mmd_squared

__all__ = ["KernelDistance"]


class KernelDistance(Metric):
    """Constructor kwargs: ``feature_extractor`` (nested ``_target_``, plain ``nn.Module``; see
    ``features.py``), ``device`` (default ``"cpu"``), ``bandwidth`` (RBF kernel bandwidth;
    default ``None`` uses the median-pairwise-distance heuristic, computed at ``compute`` time
    since it needs the full pooled sample)."""

    def __init__(self, feature_extractor: Any, device: str = "cpu", bandwidth: float | None = None) -> None:
        self.bandwidth = bandwidth
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
        mmd_squared = round(rbf_mmd_squared(self._pred.embeddings(), self._real.embeddings(), bandwidth=self.bandwidth), 6)
        return {
            "n_pred": self._pred.n,
            "n_real": self._real.n,
            "mmd_squared": mmd_squared,
            # Pre-shaped for Table's `rows: list[dict]` kwarg.
            "rows": [{"metric": "mmd_squared", "value": mmd_squared, "n_pred": self._pred.n, "n_real": self._real.n}],
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
