"""
Axis-aligned box IoU kernel.

Source:
  - title: "The PASCAL Visual Object Classes (VOC) Challenge"
    url: "https://doi.org/10.1007/s11263-009-0275-4"
    year: 2010

Description:
  A kernel is a pure sub-part of a node, never a graph node itself. It has no ``in``/``out`` and
  is passed as a nested ``_target_`` constructor kwarg (``similarity: {_target_: ...IouXyxy}`` on
  ``InstanceMatch``). See packages/teia/specs/base/eval/nodes.md §3/§5.
"""

from __future__ import annotations

from typing import Sequence

from teia.node.eval.kernel._geometry import bbox_iou, bbox_to_xyxy

__all__ = ["IouXyxy"]


class IouXyxy:
    """``pred``/``gt``: ``(cx, cy, w, h)``, any consistent unit (this codebase's capture store
    uses fractional 0-1)."""

    def __call__(self, pred: Sequence[float], gt: Sequence[float]) -> float:
        pred_xyxy = bbox_to_xyxy(float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]))
        gt_xyxy = bbox_to_xyxy(float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3]))
        return bbox_iou(pred_xyxy, gt_xyxy)
