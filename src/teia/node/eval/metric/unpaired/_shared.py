"""
Unpaired distributional distances: Fréchet, MMD and Kolmogorov-Smirnov.

Source:
  - title: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    url: "https://arxiv.org/abs/1706.08500"
    year: 2017
  - title: "The Fréchet distance between multivariate normal distributions"
    url: "https://doi.org/10.1016/0047-259X(82)90077-X"
    year: 1982
  - title: "A Kernel Two-Sample Test"
    url: "https://jmlr.org/papers/v13/gretton12a.html"
    year: 2012
  - title: "Table for Estimating the Goodness of Fit of Empirical Distributions"
    url: "https://doi.org/10.1214/aoms/1177730256"
    year: 1948

Description:
  Pure math shared by the unpaired metrics. ``frechet_distance`` is the FID closed form, whose
  Gaussian-to-Gaussian expression is due to Dowson & Landau and whose use as a generative-model
  score is due to Heusel et al. ``rbf_mmd_squared`` is Gretton et al.'s unbiased two-sample MMD
  estimator; ``ks_statistic`` is the Smirnov two-sample statistic.

Adaptations:
  - Not a component itself (no ``_target_``, no ``in``/``out``); imported by the metric modules
    in this package. Numpy-only, no scipy, so ``teia``'s base install stays light.
  - Squared distances use the Gram identity rather than an ``(N, M, D)`` broadcast difference,
    which keeps peak memory at ``(N, M)`` instead of ``(N, M, D)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frechet_distance", "rbf_mmd_squared", "ks_statistic"]


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition (no ``scipy.linalg.sqrtm``).
    Negative eigenvalues from floating-point noise are clipped to 0 before the square root."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.T


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances via ``|x|^2 + |y|^2 - 2*x@y.T``.

    The Gram identity holds the result at ``(N, M)``; the equivalent broadcast difference would
    allocate ``(N, M, D)``, which is what makes this the memory-critical helper for embedding
    sets of realistic width. Cancellation can leave small negatives on near-identical rows, so
    the result is clipped at 0.
    """
    return np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * (x @ y.T),
        0.0,
    )


def frechet_distance(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Fréchet distance between two Gaussians ``N(mean_a, cov_a)`` and ``N(mean_b, cov_b)`` fit
    to two embedding sets (the FID formula):
    ``||mu_a - mu_b||^2 + Tr(C_a + C_b - 2*sqrt(C_a @ C_b))``.

    The cross term uses the symmetric-PSD-safe square root of ``C_a^{1/2} @ C_b @ C_a^{1/2}``,
    which is better-conditioned than ``sqrtm(C_a @ C_b)`` and always real for PSD inputs,
    avoiding the complex-valued noise ``sqrtm`` of a non-symmetric product can produce.
    """
    mean_diff_sq = float(np.sum((mean_a - mean_b) ** 2))
    sqrt_cov_a = _matrix_sqrt_psd(cov_a)
    cross = _matrix_sqrt_psd(sqrt_cov_a @ cov_b @ sqrt_cov_a)
    trace_term = float(np.trace(cov_a) + np.trace(cov_b) - 2.0 * np.trace(cross))
    return max(mean_diff_sq + trace_term, 0.0)


def rbf_mmd_squared(features_a: np.ndarray, features_b: np.ndarray, *, bandwidth: float | None = None) -> float:
    """Squared Maximum Mean Discrepancy between two embedding sets under an RBF kernel
    ``k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))``. ``bandwidth`` defaults to the median
    pairwise distance across the pooled sample (the standard "median heuristic")."""
    pooled = np.concatenate([features_a, features_b], axis=0)
    if bandwidth is None:
        pairwise = np.sqrt(_squared_distances(pooled, pooled))
        np.fill_diagonal(pairwise, 0.0)
        # Coincident rows are excluded from the median. The Gram identity resolves a duplicate
        # pair as fp noise rather than exact 0, and the sqrt lifts that residual to ~sqrt(eps)
        # of the scale, so the cut is a relative tolerance rather than `> 0`.
        nonzero = pairwise[pairwise > 1e-6 * max(float(pairwise.max()), 1.0)]
        bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    bandwidth = max(float(bandwidth), 1e-12)

    def _kernel_mean(x: np.ndarray, y: np.ndarray, *, exclude_diagonal: bool) -> float:
        kernel = np.exp(-_squared_distances(x, y) / (2.0 * bandwidth**2))
        if exclude_diagonal and x.shape[0] == y.shape[0]:
            np.fill_diagonal(kernel, 0.0)
            denom = x.shape[0] * (x.shape[0] - 1)
            return float(kernel.sum() / denom) if denom > 0 else 0.0
        return float(kernel.mean())

    k_aa = _kernel_mean(features_a, features_a, exclude_diagonal=True)
    k_bb = _kernel_mean(features_b, features_b, exclude_diagonal=True)
    k_ab = _kernel_mean(features_a, features_b, exclude_diagonal=False)
    return max(k_aa + k_bb - 2.0 * k_ab, 0.0)


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two samples'
    empirical CDFs, over one 1D array each. No p-value (that needs the Kolmogorov distribution
    from scipy); the statistic alone is a bounded ``[0, 1]`` distance, which is what
    :class:`.distribution_test.DistributionTest` reports."""
    combined = np.sort(np.concatenate([sample_a, sample_b]))
    cdf_a = np.searchsorted(np.sort(sample_a), combined, side="right") / max(len(sample_a), 1)
    cdf_b = np.searchsorted(np.sort(sample_b), combined, side="right") / max(len(sample_b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b))) if combined.size else 0.0
