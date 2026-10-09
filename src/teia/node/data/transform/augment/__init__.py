"""Train-only vision augment nodes over the letterboxed item workspace.

Source: common knowledge

Description:
  ``augment``-phase ``teia.base.data.Transform`` nodes that rewrite the deterministic letterbox item keys
  (``item.image`` + normalized-cxcywh ``item.boxes`` / ``item.keypoints`` / ``item.polys`` / ``item.box_cls``).
  The executor runs them on the train split only and skips them on eval; because every node is an in-place
  rewrite, skipping leaves the letterbox values for the downstream raster/collate consumers. ``Mosaic``
  needs sibling items and so declares ``needs_siblings``; the executor injects a ``siblings`` provider.
"""

from __future__ import annotations

from typing import Any

from teia.base.data import Transform


def _suffix(key: str) -> str:
    return key.split(".", 1)[1]


class ItemAugment(Transform):
    """Base for augment nodes: maps positional ``in_key`` values to a modality record, applies ``apply``,
    and returns the ``out_key`` slice (a rewrite, so ``out_key`` ⊆ ``in_key``).

    A modality record is keyed by the ``item.``-stripped suffix — ``image``, ``boxes``, ``box_cls``,
    ``keypoints``, ``polys`` — so one node serves det/pose/seg/obb by wiring only the keys a task carries.
    """

    phase = "augment"
    needs_siblings: bool = False

    def __call__(self, *inputs: Any) -> Any:
        record = {_suffix(key): value for key, value in zip(self.in_key, inputs)}
        record = self.apply(record)
        outputs = [record[_suffix(key)] for key in self.out_key]
        return tuple(outputs) if len(outputs) != 1 else outputs[0]

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _sibling_record(self, workspace: dict[str, Any]) -> dict[str, Any]:
        return {_suffix(key): workspace[key] for key in self.in_key if key in workspace}


from teia.node.data.transform.augment.affine import RandomPerspectiveCrop
from teia.node.data.transform.augment.box_hygiene import ClampBoxes, SanitizeBoxes
from teia.node.data.transform.augment.color import ColorJitter
from teia.node.data.transform.augment.flip import RandomHFlip
from teia.node.data.transform.augment.mosaic import Mosaic, tick_mosaic_schedule
from teia.node.data.transform.augment.spec_augment import SpecAugment

__all__ = [
    "ClampBoxes",
    "ColorJitter",
    "ItemAugment",
    "Mosaic",
    "RandomHFlip",
    "RandomPerspectiveCrop",
    "SanitizeBoxes",
    "SpecAugment",
    "tick_mosaic_schedule",
]
