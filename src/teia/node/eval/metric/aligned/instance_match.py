"""
InstanceMatch: one matcher for det/obb/inst-seg/pose, parameterized by a similarity kernel.

Source:
  - title: "The PASCAL Visual Object Classes (VOC) Challenge"
    url: "https://doi.org/10.1007/s11263-009-0275-4"
    year: 2010
  - title: "Microsoft COCO: Common Objects in Context"
    url: "https://arxiv.org/abs/1405.0312"
    year: 2014

Description:
  The per-image greedy rank-then-match algorithm these four routes share: rank a route's
  predictions by score, and within each image, greedily assign each prediction (highest score
  first) to the best-overlapping not-yet-claimed ground-truth instance of the same category,
  under a similarity-kernel threshold. The matcher runs once per threshold in ``overlap_thresholds``
  (COCO's ``0.50:0.05:0.95`` by default in the shipped leaf). Output rows are
  ``{score, category, is_true_positive, threshold}`` plus the ground-truth count per category, which is what
  :class:`~teia.node.eval.metric.aligned.ranked_average_precision.RankedAveragePrecision` consumes
  via its ``matched`` input, so there is no separate detection-AP metric.

Column access instead of ``in``/``out`` wiring
----------------------------------------------
Predicted and ground-truth atoms share trailing names (``sample_idx``/``batch_idx``, ``boxes``), which
``runner._atom_name``'s kwarg naming would collide, so the metric names its columns explicitly in
``pred``/``gt`` maps and reads them via ``source.column(...)``. Its ``in`` list still declares every
column so graph wiring and contract validation cover them. Every geometry row is aligned with its
category/score/index row (the task contract's ragged atoms share one index).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from teia.base.eval import Metric

__all__ = ["InstanceMatch"]


class InstanceMatch(Metric):
    """Constructor kwargs (evalmodule config):

    - ``similarity``: a nested kernel ``_target_`` (``IouXyxy``/``IouRotated``/``IouMask``/``Oks``),
      called as ``similarity(pred_geometry, gt_geometry) -> float``.
    - ``pred``: ``{geometry, category, score, index}`` capture keys.
    - ``gt``: ``{geometry, category, index}`` target keys.
    - ``overlap_thresholds``: minimum similarities to count as a match; one matching pass per value
      (default ``[0.5]``).

    ``from_source`` returns ``rows`` (one per prediction per threshold), ``ground_truth`` (instances per
    category, matched or not, which is the recall denominator) and the totals.
    """

    def __init__(self, similarity: Any, pred: dict[str, str], gt: dict[str, str], overlap_thresholds: Sequence[float] = (0.5,)) -> None:
        self.similarity = similarity
        self.pred = dict(pred)
        self.gt = dict(gt)
        self.overlap_thresholds = tuple(float(t) for t in overlap_thresholds)
        self.reset()

    # Streaming: matching groups by (sample index, category), strictly per-image, so matching one
    # batch at a time and accumulating rows equals matching the whole split. ``_source`` is the
    # reserved kwarg the streaming runner hands a metric that reads columns directly.

    def update(self, *, _source: Any = None, **_: Any) -> None:
        if _source is None:
            return
        batch = self.from_source(_source)
        self._buffer.extend(batch["rows"])
        self._ground_truth.update(batch["ground_truth"])

    def compute(self) -> dict[str, Any]:
        return {
            "rows": list(self._buffer),
            "ground_truth": dict(self._ground_truth),
            "n_predictions": len(self._buffer) // len(self.overlap_thresholds),
            "n_ground_truth": sum(self._ground_truth.values()),
        }

    def reset(self) -> None:
        self._buffer: list[dict[str, Any]] = []
        self._ground_truth: Counter[int] = Counter()

    def from_source(self, source: Any, **_: Any) -> dict[str, Any]:
        pred_rows = self._instances(source, self.pred, scored=True)
        gt_rows = self._instances(source, self.gt, scored=False)
        return {
            "rows": self._match(pred_rows, gt_rows),
            "ground_truth": dict(Counter(row["category"] for row in gt_rows)),
            "n_predictions": len(pred_rows),
            "n_ground_truth": len(gt_rows),
        }

    def _instances(self, source: Any, keys: dict[str, str], *, scored: bool) -> list[dict[str, Any]]:
        """One row per instance: ``{sample_idx, geometry, category, score}``."""
        geometry = list(source.column(keys["geometry"]))
        category = _flatten(source.column(keys["category"]))
        index = _flatten(source.column(keys["index"]))
        score = _flatten(source.column(keys["score"])) if scored else [0.0] * len(geometry)
        return [
            {"sample_idx": int(idx), "geometry": geom, "category": int(cat), "score": float(sc)}
            for geom, cat, sc, idx in zip(geometry, category, score, index, strict=True)
        ]

    # -- matching ----------------------------------------------------------------------------

    def _match(self, pred_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        gt_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in gt_rows:
            gt_by_key.setdefault((row["sample_idx"], row["category"]), []).append(row)

        ranked = sorted(pred_rows, key=lambda row: row["score"], reverse=True)
        overlaps = [
            [float(self.similarity(pred["geometry"], gt["geometry"])) for gt in gt_by_key.get((pred["sample_idx"], pred["category"]), [])]
            for pred in ranked
        ]
        matched: list[dict[str, Any]] = []
        for threshold in self.overlap_thresholds:
            claimed: dict[tuple[int, int], set[int]] = {}
            for pred, candidate_overlaps in zip(ranked, overlaps, strict=True):
                taken = claimed.setdefault((pred["sample_idx"], pred["category"]), set())
                best_overlap, best_index = 0.0, -1
                for index, overlap in enumerate(candidate_overlaps):
                    if index not in taken and overlap > best_overlap:
                        best_overlap, best_index = overlap, index
                is_true_positive = best_index >= 0 and best_overlap >= threshold
                if is_true_positive:
                    taken.add(best_index)
                matched.append({"score": pred["score"], "category": pred["category"], "is_true_positive": is_true_positive, "threshold": threshold})
        return matched


def _flatten(column: Any) -> list[Any]:
    """A per-instance scalar column (``[N]``, ``[N, 1]``, store memmap or live tensor) as a flat list."""
    return [] if column is None else np.asarray(column).reshape(-1).tolist()
