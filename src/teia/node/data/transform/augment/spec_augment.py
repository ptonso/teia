"""SpecAugment — random time/frequency masking over the item log-mel spectrogram.

Source:
  - title: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    url: "https://arxiv.org/abs/1904.08779"
    year: 2019

Description:
  Zero-cost spectrogram augmentation: masks random frequency bands and time steps of
  ``item.audio`` with the spectrogram mean. Time warping from the paper is omitted (its gain is
  marginal next to masking and it needs a differentiable warp).
"""

from __future__ import annotations

import random
from typing import Any

from teia.node.data.transform.augment import ItemAugment


class SpecAugment(ItemAugment):
    """Mask random frequency/time bands of the ``[1, M, F]`` spectrogram with probability ``p``."""

    def __init__(
        self,
        *,
        p: float = 1.0,
        freq_mask_param: int = 12,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        self.p = float(p)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.n_freq_masks = int(n_freq_masks)
        self.n_time_masks = int(n_time_masks)

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return record
        spec = record["audio"].clone()
        n_mels, n_frames = spec.shape[-2], spec.shape[-1]
        fill = spec.mean()
        for _ in range(self.n_freq_masks):
            width = random.randint(0, min(self.freq_mask_param, n_mels))
            if width:
                start = random.randint(0, n_mels - width)
                spec[..., start : start + width, :] = fill
        for _ in range(self.n_time_masks):
            width = random.randint(0, min(self.time_mask_param, n_frames))
            if width:
                start = random.randint(0, n_frames - width)
                spec[..., :, start : start + width] = fill
        record["audio"] = spec
        return record
