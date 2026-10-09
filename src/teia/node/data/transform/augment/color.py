"""Photometric color-jitter augment over the item workspace.

Source: common knowledge

Description:
  ``ItemAugment`` wrapper around ``torchvision.transforms.v2.ColorJitter`` — a bare torchvision class
  cannot be wired as a flat data-graph node directly (it does not satisfy the
  ``Reader/Transform/Join/Reshape/Collate/Writer`` ABC inference the graph builder requires), so this
  thin wrapper gives it a node identity. Image-only; train-phase (``phase="augment"``, skipped on eval).
"""

from __future__ import annotations

from typing import Any

from teia.core.deps import require_dependency
from teia.node.data.transform.augment import ItemAugment


class ColorJitter(ItemAugment):
    """``item.image`` in-place photometric jitter (brightness/contrast/saturation/hue)."""

    def __init__(
        self,
        *,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._jitter: Any = None

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        if self._jitter is None:
            require_dependency("torchvision", "ColorJitter")
            from torchvision.transforms import v2

            self._jitter = v2.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
        record["image"] = self._jitter(record["image"])
        return record
