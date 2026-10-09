"""
Polygon-overlap IoU kernel.

Source:
  - title: "Microsoft COCO: Common Objects in Context"
    url: "https://arxiv.org/abs/1405.0312"
    year: 2014
  - title: "Reentrant Polygon Clipping"
    url: "https://doi.org/10.1145/360767.360802"
    year: 1974

Description:
  A kernel is a pure sub-part of a node, never a graph node itself. See
  packages/teia/specs/base/eval/nodes.md §3/§5.

Named ``iou_mask`` per the kernel taxonomy, but it operates on polygon point lists, not raster
masks. The capture store writes instance-segmentation ground truth and predictions as polygons
(``capture.inst-seg.polys`` / ``batch.polys``), never rasterized blobs, so an analytic
polygon-clipping IoU is what the data supports, and it is exact where a rasterize-then-count
approach would introduce discretization error. A genuine raster-mask IoU, for a route that
captures blobs instead, would be a different kernel reading the blob lane.
"""

from __future__ import annotations

from typing import Sequence

from teia.node.eval.kernel._geometry import polygon_iou

__all__ = ["IouMask"]


class IouMask:
    """``pred``/``gt``: ``[(x, y), ...]`` polygon point lists, fractional 0-1. Exact only for
    convex polygons (Sutherland-Hodgman clipping)."""

    def __call__(self, pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]]) -> float:
        pred_points = [(float(x), float(y)) for x, y in pred]
        gt_points = [(float(x), float(y)) for x, y in gt]
        return polygon_iou(pred_points, gt_points)
