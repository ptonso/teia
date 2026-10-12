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
  nothing about the distribution shape. The estimate is averaged over random subset pairs, which
  bounds memory at ``subset_size^2``. With ``kernel: poly3`` (``(x.y / d + 1)^3``) and Inception
  pool3 features this is Binkowski et al.'s KID; ``kernel: rbf`` uses the median-distance
  bandwidth inside each subset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from teia.base.eval import Metric
from teia.node.eval.metric.unpaired._embedding import EmbeddingAccumulator
from teia.node.eval.metric.unpaired._shared import poly3_mmd_squared, rbf_mmd_squared

__all__ = ["KernelDistance"]

KERNELS = ("poly3", "rbf")


class KernelDistance(Metric):
    """Subset-averaged unbiased MMD^2 (Gretton et al. 2012; KID with ``poly3``, Binkowski et al. 2018).

    Args:
        feature_extractor: Nested ``_target_``, a plain ``nn.Module`` (see ``features.py``).
        device: Extractor device.
        kernel: ``poly3`` or ``rbf``.
        subsets: Number of random subset pairs averaged.
        subset_size: Rows per subset, capped by the smaller set.
        bandwidth: RBF bandwidth; ``None`` takes the median pairwise distance of each subset.
        seed: Subset sampling seed.
        batch_size: Images per extractor call.
    """

    def __init__(
        self,
        feature_extractor: Any,
        device: str = "cpu",
        kernel: str = "poly3",
        subsets: int = 100,
        subset_size: int = 1000,
        bandwidth: float | None = None,
        seed: int = 0,
        batch_size: int = 256,
    ) -> None:
        if kernel not in KERNELS:
            raise ValueError(f"KernelDistance: kernel {kernel!r} is not one of {KERNELS}.")
        self.kernel, self.subsets, self.subset_size = kernel, int(subsets), int(subset_size)
        self.bandwidth, self.seed = bandwidth, int(seed)
        self._pred = EmbeddingAccumulator(feature_extractor, device=device, batch_size=batch_size)
        self._real = EmbeddingAccumulator(feature_extractor, device=device, batch_size=batch_size)

    def update(self, *, pred: torch.Tensor | None = None, real: torch.Tensor | None = None, **_: Any) -> None:
        if pred is not None:
            self._pred.update_tensors(pred)
        if real is not None:
            self._real.update_tensors(real)

    def compute(self) -> dict[str, Any]:
        if self._pred.n == 0 or self._real.n == 0:
            raise ValueError(f"KernelDistance: needs predicted and real images, got {self._pred.n} and {self._real.n}.")
        pred, real = self._pred.embeddings(), self._real.embeddings()
        size = min(self.subset_size, len(pred), len(real))
        rng = np.random.default_rng(self.seed)
        values = [
            self._mmd(pred[rng.choice(len(pred), size, replace=False)], real[rng.choice(len(real), size, replace=False)])
            for _ in range(self.subsets)
        ]
        mean, std = round(float(np.mean(values)), 6), round(float(np.std(values)), 6)
        return {
            "n_pred": self._pred.n,
            "n_real": self._real.n,
            "mmd_squared": mean,
            "mmd_squared_std": std,
            # Pre-shaped for Table's `rows: list[dict]` kwarg.
            "rows": [{"metric": f"mmd_squared_{self.kernel}", "value": mean, "std": std, "n_pred": self._pred.n, "n_real": self._real.n}],
        }

    def _mmd(self, a: np.ndarray, b: np.ndarray) -> float:
        return poly3_mmd_squared(a, b) if self.kernel == "poly3" else rbf_mmd_squared(a, b, bandwidth=self.bandwidth)

    def reset(self) -> None:
        self._pred.reset()
        self._real.reset()

    def from_source(self, source: Any, *, pred_paths: list[str] | None = None, real_paths: list[str] | None = None, **_: Any) -> dict[str, Any]:
        del source
        self.reset()
        self._pred.update_paths(list(pred_paths or []))
        self._real.update_paths(list(real_paths or []))
        return self.compute()
