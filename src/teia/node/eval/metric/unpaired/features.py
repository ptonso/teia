"""
Plain ``nn.Module`` feature extractors for the unpaired metrics' nested ``feature_extractor``.

Source: common knowledge

Description:
  constructor kwarg (see ``frechet_distance.py``/``kernel_distance.py``).

Not a ``TeiaNode`` subclass. ``run_eval_graph`` instantiates a node's config via plain,
recursive ``instantiate``, so a nested ``{_target_: ...}`` kwarg resolves automatically for an
ordinary class. A ``TeiaNode`` would additionally need ``in_shape``/``out_shape`` injected by
``teia.core.module.net``'s shape-inference machinery, which the eval graph has no analog of, so
a feature extractor referenced here must be a plain module that knows its own input shape.

Not a component in the ``metric``/``view``/``compare``/``kernel`` taxonomy either: it is a
nested sub-part of a ``Metric``, the same relationship a kernel has to ``InstanceMatch``, but
not a kernel (a kernel is a pure similarity function of two already-extracted values; this is a
learned embedding step upstream of that).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FlattenFeatures"]


class FlattenFeatures(nn.Module):
    """Global-average-pool and flatten: maps a ``(N, C, H, W)`` image batch to a ``(N, C)``
    embedding.

    Not a perceptual feature extractor: no learned weights, no Inception/CLIP-grade semantics. A
    deployment wanting FID-grade numbers wires a pretrained network's ``_target_`` here instead
    (any ``nn.Module`` with a matching ``forward`` signature works). This is the light,
    dependency-free default: exact, deterministic, and enough to exercise the unpaired-metric
    mechanism end to end.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return images.mean(dim=(2, 3))
