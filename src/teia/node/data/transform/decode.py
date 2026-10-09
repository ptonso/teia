"""``DecodeImage``/``DecodeAudio`` — read media files to tensors (item/preprocess).

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.core.deps import require_dependency
from teia.base.data import Transform


class DecodeImage(Transform):
    """``item.img_path → item.image`` (uint8 ``[3,H,W]`` RGB). Float conversion happens post-transfer."""

    @staticmethod
    def kernel(value: str, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        from torchvision.io import ImageReadMode, read_image

        return read_image(str(value), mode=ImageReadMode.RGB), {}


class DecodeImageWithOriShape(Transform):
    """``item.img_path → (item.image, item.ori_shape)``. Same decode as :class:`DecodeImage`, plus
    the pre-resize ``(H, W)`` as a second output.

    For grid tasks with no letterbox transform (sem-seg): a later ``item.image → item.image``
    ``Resize`` overwrites the key, and the data graph's producer resolution binds any *plain*
    consumer of a rewritten key to its *final* version (``teia/core/datamodule/graph.py``'s
    ``producer_map``), not an intermediate one — so a separate node reading ``item.image`` to
    capture its shape would always see the post-resize size, never the original. Emitting
    ``ori_shape`` as a second output from the same node that first produces ``item.image``
    sidesteps that: it is a plain, single-producer key with no rewrite chain of its own.
    """

    #: The export/dry-run kernel-chain walker (``core/export/preprocess_slice.py``) only traces
    #: the ``image`` field, so this is the plain single-output decode kernel; ``ori_shape`` is
    #: an eval/report-time concern, never part of an exported inference chain.
    kernel = DecodeImage.kernel

    def __call__(self, path: str) -> tuple[torch.Tensor, tuple[int, int]]:
        image, _ = DecodeImage.kernel(path, {})
        return image, (int(image.shape[-2]), int(image.shape[-1]))


class DecodeAudio(Transform):
    """``item.audio_path → item.waveform`` (float32 ``[1,T]`` mono at ``sample_rate``).

    Decodes via ``soundfile`` (stable across encoded formats without an ffmpeg/GPU-backend
    dependency), resamples to ``sample_rate`` via ``torchaudio.functional.resample``, downmixes to
    mono, and (when ``duration`` is set) crops/zero-pads to a fixed window so a plain ``Stack``
    collate suffices. Precomputed feature files (``.npy``/``.pt``) bypass decoding and are loaded
    as-is.
    """

    def __init__(self, *, sample_rate: int = 16000, mono: bool = True, duration: float | None = None) -> None:
        require_dependency("soundfile", "DecodeAudio")
        require_dependency("torchaudio", "DecodeAudio")
        self.sample_rate = int(sample_rate)
        self.mono = bool(mono)
        self.duration = float(duration) if duration is not None else None

    def params(self) -> dict[str, Any]:
        return {"sample_rate": self.sample_rate, "mono": self.mono, "duration": self.duration}

    @staticmethod
    def kernel(value: str, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        path = str(value)
        if path.endswith(".pt"):
            return torch.load(path, weights_only=True), {}
        if path.endswith(".npy"):
            import numpy as np

            return torch.from_numpy(np.load(path)), {}

        import soundfile as sf
        import torchaudio

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.T)  # [frames, channels] -> [C, T]
        sample_rate = int(params["sample_rate"])
        if int(sr) != sample_rate:
            wave = torchaudio.functional.resample(wave, int(sr), sample_rate)
        if params["mono"] and wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        duration = params["duration"]
        if duration is not None:
            length = int(round(float(duration) * sample_rate))
            if wave.shape[-1] >= length:
                wave = wave[..., :length]
            else:
                wave = torch.nn.functional.pad(wave, (0, length - wave.shape[-1]))
        return wave, {}


class DecodeMask(Transform):
    """``item.mask_path → item.mask`` (int64 ``[H,W]`` class-index semantic mask)."""

    def __call__(self, path: str | None) -> torch.Tensor:
        if path is None:
            return torch.zeros((0, 0), dtype=torch.int64)
        require_dependency("torchvision", "DecodeMask")
        from torchvision.io import ImageReadMode, read_image

        return read_image(str(path), mode=ImageReadMode.GRAY)[0].to(torch.int64)
