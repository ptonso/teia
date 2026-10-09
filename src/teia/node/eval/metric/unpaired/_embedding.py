"""
Shared embedding-accumulation plumbing for the unpaired metrics. Not a component itself.

Source: common knowledge

Description:
  The eval graph runs offline, with no ``LightningModule`` or trainer, so the inference hygiene an
  unpaired metric needs (``.eval()``, ``torch.no_grad()``, device placement) is this package's own
  responsibility. See ``features.py`` for why the feature extractor itself must be a plain
  ``nn.Module`` rather than a ``TeiaNode``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

__all__ = ["EmbeddingAccumulator", "load_image_tensor"]


def load_image_tensor(path: str) -> torch.Tensor:
    """One image, ``(C, H, W)`` float32 in ``[0, 1]``: the convention the ``features.py``
    extractors and the capture store's raster paths (``capture.gen.image_path``) agree on."""
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class EmbeddingAccumulator:
    """Runs a batch of images through a feature extractor and accumulates the resulting
    embeddings. Used identically by the streaming and materialized drivers of both
    :class:`.frechet_distance.FrechetDistance` and :class:`.kernel_distance.KernelDistance`, so
    the inference hygiene (``.eval()``, ``torch.no_grad()``, device placement) and image-batch
    construction live in one place."""

    def __init__(self, feature_extractor: Any, *, device: str = "cpu") -> None:
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self._embeddings: list[np.ndarray] = []

    def reset(self) -> None:
        self._embeddings = []

    def update_tensors(self, images: torch.Tensor) -> None:
        with torch.no_grad():
            batch = images.to(self.device)
            embedded = self.feature_extractor(batch)
        embedded = embedded.detach()
        if embedded.dtype == torch.bfloat16:
            # numpy has no native bfloat16 dtype; widen before conversion.
            embedded = embedded.float()
        self._embeddings.append(embedded.cpu().numpy())

    def update_paths(self, paths: list[str], *, batch_size: int = 32) -> None:
        valid_paths = [path for path in paths if path]
        for start in range(0, len(valid_paths), batch_size):
            chunk = valid_paths[start : start + batch_size]
            images = torch.stack([load_image_tensor(path) for path in chunk])
            self.update_tensors(images)

    @property
    def n(self) -> int:
        return sum(array.shape[0] for array in self._embeddings)

    def embeddings(self) -> np.ndarray:
        if not self._embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(self._embeddings, axis=0)

    def mean_and_cov(self) -> tuple[np.ndarray, np.ndarray]:
        features = self.embeddings()
        mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        return mean, cov
