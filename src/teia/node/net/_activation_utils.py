"""
Shared Activation Decode Utilities.

Source: common knowledge

Description:
  Provides decode math shared by 2+ activation kernels (single-label, multi-label, regression, policy, grid detection).
  Each function is a pure ``(activated, ctx) -> list[dict]`` callable so an activation's ``kernel``
  staticmethod can delegate to it and have the call collected verbatim into an export bundle.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from teia.core.export.kernels import to_numpy


def decode_regression(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Per-sample ``{"value": [T]}`` rows from activated logits."""
    del ctx
    values = to_numpy(activated["value"])
    return [{"value": row} for row in values.reshape(values.shape[0] if values.ndim else 1, -1)]


def decode_policy(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Per-sample ``{"action": [A]}`` rows from deterministic actions."""
    del ctx
    action = to_numpy(activated["action"])
    return [{"action": row} for row in action.reshape(action.shape[0] if action.ndim else 1, -1)]


def decode_multilabel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Multi-label decode: ``scores`` and multi-hot ``labels`` thresholded per class by ``ctx.thresholds``."""
    probs = to_numpy(activated["probs"])
    probs = probs[None, :] if probs.ndim == 1 else probs
    resolved = np.asarray(resolve_thresholds(ctx.get("thresholds"), probs.shape[-1], float(ctx.get("fallback_threshold", 0.5))))
    return [{"scores": row, "labels": (row >= resolved).astype(np.int64)} for row in probs]


def decode_singlelabel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Single-label decode: ``scores`` and the ``label`` of ``argmax(ctx.weights * scores)`` (plain argmax without weights)."""
    probs = to_numpy(activated["probs"])
    probs = probs[None, :] if probs.ndim == 1 else probs
    weights = np.asarray(resolve_thresholds(ctx.get("weights"), probs.shape[-1], 1.0))
    return [{"scores": row, "label": int((row * weights).argmax())} for row in probs]


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression of ``[N, 4]`` ``xyxy`` boxes, best score first."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        best, order = order[0], order[1:]
        keep.append(best)
        if not order.size:
            break
        inter = np.clip(np.minimum(boxes[best, 2], boxes[order, 2]) - np.maximum(boxes[best, 0], boxes[order, 0]), 0, None) * np.clip(
            np.minimum(boxes[best, 3], boxes[order, 3]) - np.maximum(boxes[best, 1], boxes[order, 1]), 0, None
        )
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = order[inter <= iou * np.maximum(areas[best] + areas[order] - inter, 1e-12)]
    return np.asarray(keep, dtype=np.int64)


def restore_xy(points: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int], ratio_pad: Any | None = None) -> np.ndarray:
    """Pixel ``[..., 2]`` points ``(x, y)`` of the model frame in the ``to_hw`` frame; ``ratio_pad`` undoes a letterbox."""
    out = np.array(points, dtype=np.float32)
    if ratio_pad is not None:
        out[..., 0] = (out[..., 0] - float(ratio_pad[1][0])) / float(ratio_pad[0][0])
        out[..., 1] = (out[..., 1] - float(ratio_pad[1][1])) / float(ratio_pad[0][0])
        return out
    out[..., 0] *= max(int(to_hw[1]), 1) / max(int(from_hw[1]), 1)
    out[..., 1] *= max(int(to_hw[0]), 1) / max(int(from_hw[0]), 1)
    return out


def frac_cxcywh(xyxy: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Pixel ``xyxy`` boxes of a ``hw`` frame as fractional ``cxcywh``, the eval and capture box convention."""
    h, w = max(int(hw[0]), 1), max(int(hw[1]), 1)
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 2] - boxes[:, 0]) / w, (boxes[:, 3] - boxes[:, 1]) / h], axis=1)


def grid_postprocess(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
    """Per-sample detections from grid candidates: confidence filter, NMS, letterbox restore, fractional atoms.

    ``activated`` holds ``boxes [B, A, 4]`` fractional ``xyxy`` of the model frame and ``scores [B, A, C]``, plus ``angles [B, A, 1]``
    (``ctx["extra"] == "angle"``) or ``keypoints [B, A, K, D]`` fractional ``(x, y, ...)`` (``"keypoints"``). With ``"polygon"`` each
    instance also carries its box rectangle as ``polys``. The frame is ``ori_shape`` when ``ctx`` has restore values, else ``image_hw``.
    """
    conf, iou, max_det = float(ctx.get("conf", 0.001)), float(ctx.get("iou", 0.7)), int(ctx.get("max_det", 300))
    from_hw = tuple(ctx["image_hw"])
    shapes, pads, extra = list(ctx.get("ori_shape") or []), list(ctx.get("ratio_pad") or []), ctx.get("extra", "none")
    boxes_b, scores_b = to_numpy(activated["boxes"]), to_numpy(activated["scores"])
    out = []
    for i in range(boxes_b.shape[0]):
        scores, labels = scores_b[i].max(axis=1), scores_b[i].argmax(axis=1)
        keep = np.flatnonzero(scores >= conf)
        keep = keep[greedy_nms(boxes_b[i][keep], scores[keep], iou)[:max_det]]
        shape = tuple(int(v) for v in shapes[i]) if i < len(shapes) else from_hw
        pad = pads[i] if i < len(pads) else None
        corners = restore_xy(boxes_b[i][keep].reshape(-1, 2, 2) * np.array([from_hw[1], from_hw[0]], dtype=np.float32), from_hw, shape, pad)
        corners = np.clip(corners, 0, np.array([shape[1], shape[0]], dtype=np.float32))
        atoms = {"boxes": frac_cxcywh(corners.reshape(-1, 4), shape), "score": scores[keep].astype(np.float32), "category": labels[keep].astype(np.int64)}
        side = np.array([shape[1], shape[0]], dtype=np.float32)
        if extra == "angle":
            atoms["boxes"] = np.concatenate([atoms["boxes"], to_numpy(activated["angles"])[i][keep]], axis=1)
        if extra == "keypoints":
            points = np.array(to_numpy(activated["keypoints"])[i][keep], dtype=np.float32)
            points[..., :2] = restore_xy(points[..., :2] * np.array([from_hw[1], from_hw[0]], dtype=np.float32), from_hw, shape, pad) / side
            atoms["keypoints"] = points
        if extra == "polygon":
            lo, hi = corners[:, 0], corners[:, 1]
            atoms["polys"] = [np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]], dtype=np.float32) / side for a, b in zip(lo, hi)]
        out.append(atoms)
    return out


def resolve_thresholds(thresholds: Any, count: int, fallback: float) -> list[float]:
    """Broadcast a scalar/list/``None`` threshold spec into one value per class."""
    if thresholds is None:
        return [float(fallback) for _ in range(count)]
    if isinstance(thresholds, (int, float)):
        return [float(thresholds) for _ in range(count)]
    values = [float(item) for item in thresholds]
    if len(values) >= count:
        return values[:count]
    return values + [float(fallback) for _ in range(count - len(values))]


class SquareFrameParams:
    """Activation mixin: publish the square model frame ``image_hw`` into the decode ctx via ``params()``.

    Kernels emit fractional coordinates of the frame they decode into; without restore values
    (capture) that frame is the model input, whose size only the datamodule knows."""

    image_size: int | None = None

    def configure_from_datamodule(self, dm: Any) -> None:
        self.image_size = int(dm.image_size)

    def params(self) -> dict[str, Any]:
        if self.image_size is None:
            raise RuntimeError(f"{type(self).__name__}: image_size unset; configure_from_datamodule must run first.")
        return {"image_hw": (self.image_size, self.image_size)}
