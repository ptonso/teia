"""
SemanticConfusion: pixel-level confusion breakdown for semantic segmentation.

Source:
  - title: "Fully Convolutional Networks for Semantic Segmentation"
    url: "https://arxiv.org/abs/1411.4038"
    year: 2015

Description:
  mIoU, Dice, pixel accuracy, frequency-weighted IoU and per-class IoU, all derived from a
  confusion matrix the same way :class:`~.confusion.Confusion` derives its metrics (diagonal over
  row+col-sum-minus-diagonal for IoU, and so on). The difference is that this matrix is
  accumulated by reading and comparing whole raster masks rather than counting per-sample class
  labels, so it is a sibling metric rather than a third mode of ``Confusion``.

Column access instead of ``in``/``out`` wiring
----------------------------------------------
Reads the ``pred`` and ``gt`` mask keys directly via ``source.column(...)``, for the same reason
``InstanceMatch`` does. A stored ``Blob`` column reads as file paths and a streaming frame holds
arrays; both are accepted. The node's ``in`` list still declares both keys for wiring validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from teia.base.eval import Metric

__all__ = ["SemanticConfusion"]


class SemanticConfusion(Metric):
    """Constructor kwargs (leaf/bundle config):

    - ``pred``/``gt``: the predicted and ground-truth mask keys (``capture.seg.mask``, ``batch.masks``).
    - ``ignore_index``: pixel value excluded from every count (default ``255``; capture atoms
      carry no per-mask ``ignore_index`` metadata).
    - ``worst_k``: how many worst off-diagonal confusion pairs and worst-mIoU samples to report
      (default ``5``).
    """

    #: ``fitness`` is mIoU, which is exactly what this metric reports, so a pipeline monitoring
    #: ``val/fitness`` for sem-seg keeps working.
    def __init__(self, pred: str, gt: str, ignore_index: int = 255, worst_k: int = 5) -> None:
        self.pred = str(pred)
        self.gt = str(gt)
        self.ignore_index = int(ignore_index)
        self.worst_k = int(worst_k)
        self.MONITORS = ("mean_iou", "fitness")
        self.reset()

    def from_source(self, source: Any, *, class_names: list[str] | None = None, sample_ids: list[str] | None = None, **_: Any) -> dict[str, Any]:
        pred_paths = list(source.column(self.pred))
        gt_paths = list(source.column(self.gt))
        resolved_classes = list(class_names) if class_names else []
        n_classes = len(resolved_classes)

        if not pred_paths or not gt_paths or n_classes == 0:
            return {"n_samples": 0}

        confusion = np.zeros((n_classes, n_classes), dtype=float)
        sample_rows: list[dict[str, Any]] = []
        total_void = 0
        total_pixels = 0

        for index, (pred_path, gt_path) in enumerate(zip(pred_paths, gt_paths, strict=False)):
            gt_mask = _mask(gt_path)
            pred_mask = _mask(pred_path)
            if gt_mask.shape != pred_mask.shape:
                raise ValueError(f"Semantic segmentation masks must have identical shapes; got {gt_mask.shape} and {pred_mask.shape}.")
            valid = gt_mask != self.ignore_index
            total_void += int((~valid).sum())
            total_pixels += int(gt_mask.size)
            if not valid.any():
                continue
            gt_values = gt_mask[valid].astype(int)
            pred_values = np.clip(pred_mask[valid].astype(int), 0, n_classes - 1)
            gt_values = np.clip(gt_values, 0, n_classes - 1)
            counts = _pixel_confusion(gt_values, pred_values, n_classes)
            confusion += counts

            diag = np.diag(counts).astype(float)
            union = counts.sum(axis=1) + counts.sum(axis=0) - diag
            ious = np.where(union > 0, diag / union, np.nan)
            sample_id = str(sample_ids[index]) if sample_ids is not None and index < len(sample_ids) else str(index)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "pixel_accuracy": round(float((gt_values == pred_values).mean()), 6),
                    "valid_pixels": int(valid.sum()),
                    "mean_iou": round(float(np.nanmean(ious)), 6),
                }
            )

        derived = _derive_from_confusion(confusion, total_void=total_void, total_pixels=total_pixels)
        iou, dice = derived["iou"], derived["dice"]
        pixel_accuracy_per_class, precision_per_class = derived["pixel_accuracy_per_class"], derived["precision_per_class"]
        prevalence, gt_pixels, normalized_confusion = derived["prevalence"], derived["gt_pixels"], derived["normalized_confusion"]

        class_rows = []
        for index, class_name in enumerate(resolved_classes):
            class_rows.append(
                {
                    "class_name": class_name,
                    "support_pixels": int(gt_pixels[index]),
                    "prevalence": None if np.isnan(prevalence[index]) else round(float(prevalence[index]), 6),
                    "iou": None if np.isnan(iou[index]) else round(float(iou[index]), 6),
                    "dice": None if np.isnan(dice[index]) else round(float(dice[index]), 6),
                    "pixel_accuracy": None if np.isnan(pixel_accuracy_per_class[index]) else round(float(pixel_accuracy_per_class[index]), 6),
                    "precision": None if np.isnan(precision_per_class[index]) else round(float(precision_per_class[index]), 6),
                }
            )

        metrics: dict[str, Any] = {"n_samples": len(sample_rows), **derived["metrics"]}

        off_diagonal = []
        for true_index, true_name in enumerate(resolved_classes):
            for pred_index, pred_name in enumerate(resolved_classes):
                if true_index == pred_index:
                    continue
                value = float(normalized_confusion[true_index, pred_index])
                if value > 0:
                    off_diagonal.append((f"{true_name}→{pred_name}", value))
        off_diagonal.sort(key=lambda item: item[1], reverse=True)
        worst_confusions = off_diagonal[: max(self.worst_k, 1)]

        sample_quality = sorted(((row["sample_id"], float(row["mean_iou"])) for row in sample_rows), key=lambda item: item[1])[: max(self.worst_k, 1)]

        return {
            **metrics,
            "class_names": resolved_classes,
            "confusion_matrix": confusion.tolist(),
            "confusion_matrix_normalized": normalized_confusion.tolist(),
            "class_rows": class_rows,
            "sample_rows": sample_rows,
            "worst_confusion_pairs": [{"pair": name, "value": value} for name, value in worst_confusions],
            "worst_samples": [{"sample_id": label, "value": value} for label, value in sample_quality],
        }

    # Streaming: accumulate a running confusion matrix directly from raw label-map tensors, since
    # no capture-store blobs exist mid-training. Reads `_source` directly rather than accepting
    # `pred_mask`/`gt_mask` as resolved kwargs, for the same reason `from_source` reads columns
    # directly (see the module docstring). The streaming live-frame builder uses its own key
    # names (raw tensors under `capture.<route>.mask` / `batch.mask`), not the report-time
    # `mask_path` columns.

    def update(self, *, _source: Any = None, class_names: list[str] | None = None, **_: Any) -> None:
        if class_names:
            self._class_names = list(class_names)
        if _source is None:
            return
        pred_mask = _source.column(self.pred)
        gt_mask = _source.column(self.gt)
        if pred_mask is None or gt_mask is None:
            return
        n_classes = len(self._class_names or [])
        if n_classes == 0:
            return
        if self._confusion is None:
            self._confusion = np.zeros((n_classes, n_classes), dtype=float)
        gt_arr = _as_numpy(gt_mask)
        pred_arr = _as_numpy(pred_mask)
        valid = gt_arr != self.ignore_index
        self._total_void += int((~valid).sum())
        self._total_pixels += int(gt_arr.size)
        if not valid.any():
            return
        gt_values = np.clip(gt_arr[valid].astype(int), 0, n_classes - 1)
        pred_values = np.clip(pred_arr[valid].astype(int), 0, n_classes - 1)
        self._confusion += _pixel_confusion(gt_values, pred_values, n_classes)

    def compute(self) -> dict[str, Any]:
        if self._confusion is None:
            return {"n_samples": 0}
        derived = _derive_from_confusion(self._confusion, total_void=self._total_void, total_pixels=self._total_pixels)
        metrics = {**derived["metrics"], "class_names": list(self._class_names or [])}
        metrics["fitness"] = metrics["mean_iou"]
        return metrics

    def reset(self) -> None:
        self._confusion: np.ndarray | None = None
        self._class_names: list[str] | None = None
        self._total_void = 0
        self._total_pixels = 0


def _as_numpy(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.dtype == torch.bfloat16:
                # numpy has no native bfloat16 dtype; widen before conversion.
                value = value.float()
            return value.cpu().numpy()
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return np.asarray(value)


def _pixel_confusion(gt_values: Any, pred_values: Any, n_classes: int) -> Any:
    encoded = (gt_values * n_classes) + pred_values
    return np.bincount(encoded, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _derive_from_confusion(confusion: Any, *, total_void: int, total_pixels: int) -> dict[str, Any]:
    """Derive every scalar and per-class quantity from a confusion matrix. Both drivers share
    this; only how the matrix is built differs (whole-split file reads vs. per-batch tensors).
    """
    diag = np.diag(confusion)
    gt_pixels = confusion.sum(axis=1)
    pred_pixels = confusion.sum(axis=0)
    union = gt_pixels + pred_pixels - diag
    iou = np.where(union > 0, diag / union, np.nan)
    dice = np.where((gt_pixels + pred_pixels) > 0, (2.0 * diag) / (gt_pixels + pred_pixels), np.nan)
    pixel_accuracy_per_class = np.where(gt_pixels > 0, diag / gt_pixels, np.nan)
    precision_per_class = np.where(pred_pixels > 0, diag / pred_pixels, np.nan)
    prevalence = gt_pixels / max(float(gt_pixels.sum()), 1.0)
    normalized_confusion = confusion / np.clip(gt_pixels[:, None], 1.0, None)
    metrics = {
        "mean_iou": round(float(np.nanmean(iou)), 6) if len(iou) else 0.0,
        "pixel_accuracy": round(float(diag.sum() / max(confusion.sum(), 1.0)), 6),
        "mean_pixel_accuracy": round(float(np.nanmean(pixel_accuracy_per_class)), 6) if len(pixel_accuracy_per_class) else 0.0,
        "frequency_weighted_iou": round(float(np.nansum(prevalence * iou)), 6) if len(prevalence) else 0.0,
        "mean_dice": round(float(np.nanmean(dice)), 6) if len(dice) else 0.0,
        "void_pixel_rate": round(float(total_void / max(total_pixels, 1)), 6),
    }
    return {
        "metrics": metrics,
        "iou": iou,
        "dice": dice,
        "pixel_accuracy_per_class": pixel_accuracy_per_class,
        "precision_per_class": precision_per_class,
        "prevalence": prevalence,
        "gt_pixels": gt_pixels,
        "normalized_confusion": normalized_confusion,
    }


def _mask(value: Any) -> np.ndarray:
    """A mask from a stored blob path or an in-memory array."""
    if isinstance(value, (str, Path)):
        return np.asarray(Image.open(Path(value)))
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
