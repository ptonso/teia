"""
Object Keypoint Similarity kernel.

Source:
  - title: "Microsoft COCO: Common Objects in Context"
    url: "https://arxiv.org/abs/1405.0312"
    year: 2014
  - title: "Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation"
    url: "https://arxiv.org/abs/1707.05388"
    year: 2017

Description:
  Object Keypoint Similarity, the ``pose`` route's similarity kernel:
  ``exp(-d^2 / (2 * s^2 * kappa^2))`` averaged over visible ground-truth keypoints, where ``d``
  is the per-keypoint distance and ``s`` the object scale. A kernel is a pure sub-part of a node,
  never a graph node itself — see [base/eval/nodes](../../../../../../../teia/specs/base/eval/nodes.md) §3/§5.

Adaptations:
  - A single uniform ``_SIGMA`` stands in for COCO's 17 per-keypoint sigmas, so every joint is
    weighted equally rather than by its annotator-variance calibration. Scores are therefore
    comparable within this repo but not directly against published COCO OKS numbers.
  - A kernel must be a pure function of ``(pred, gt)`` with no third input, so the object scale
    is derived here from the visible ground-truth keypoints' own bounding-box diagonal rather
    than from a separately-passed ground-truth bounding box.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = ["Oks"]

_SIGMA = 0.05


class Oks:
    """``pred``/``gt``: ``[(x, y, visibility), ...]``, fractional 0-1. Visibility <= 0 marks a
    ground-truth keypoint absent (not scored); a missing predicted point at a visible index
    scores as ``(0, 0, 0)``."""

    def __call__(self, pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]]) -> float:
        gt_points = [tuple(float(v) for v in point) for point in gt]
        pred_points = [tuple(float(v) for v in point) for point in pred]
        scale = max(_bbox_diagonal(gt_points), 1e-6)

        values: list[float] = []
        for index, gt_point in enumerate(gt_points):
            if len(gt_point) < 3 or gt_point[2] <= 0:
                continue
            pred_point = pred_points[index] if index < len(pred_points) else (0.0, 0.0, 0.0)
            distance = math.hypot(pred_point[0] - gt_point[0], pred_point[1] - gt_point[1])
            values.append(math.exp(-((distance**2) / (2.0 * ((_SIGMA * scale) ** 2)))))
        return float(sum(values) / len(values)) if values else 0.0


def _bbox_diagonal(points: list[tuple[float, ...]]) -> float:
    visible = [(p[0], p[1]) for p in points if len(p) >= 3 and p[2] > 0]
    if not visible:
        return 1.0
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
