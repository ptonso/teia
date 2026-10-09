"""
Rotated box IoU kernel.

Source:
  - title: "DOTA: A Large-scale Dataset for Object Detection in Aerial Images"
    url: "https://arxiv.org/abs/1711.10398"
    year: 2018

Description:
  A kernel is a pure sub-part of a node, never a graph node itself. See
  packages/teia/specs/base/eval/nodes.md §3/§5.
"""

from __future__ import annotations

from typing import Sequence

from teia.node.eval.kernel._geometry import polygon_iou, rotated_box_corners

__all__ = ["IouRotated"]


class IouRotated:
    """``pred``/``gt``: ``(cx, cy, w, h, angle)``; angle in radians, counter-clockwise, the
    convention ``item.obb`` and the OBB capture write path use. Converts each box to its 4
    corners and reuses the same analytic polygon-intersection IoU as
    :class:`.iou_mask.IouMask`."""

    def __call__(self, pred: Sequence[float], gt: Sequence[float]) -> float:
        pred_corners = rotated_box_corners(*(float(v) for v in pred[:5]))
        gt_corners = rotated_box_corners(*(float(v) for v in gt[:5]))
        return polygon_iou(pred_corners, gt_corners)
