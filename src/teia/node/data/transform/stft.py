"""``MelSpectrogram`` — waveform to log-mel spectrogram grid (item/preprocess).

Source:
  - title: "librosa: Audio and Music Signal Analysis in Python"
    url: "https://librosa.org"
    year: 2015

Description:
  Standard log-mel front end: STFT magnitude → mel filterbank → log. Computed via
  ``torchaudio.transforms.MelSpectrogram``; turns the ``sequence`` waveform into the ``grid``
  ``audio`` field so any grid encoder (convnets) runs on it unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.core.deps import require_dependency
from teia.base.data import Transform


class MelSpectrogram(Transform):
    """``item.waveform → item.audio`` (float32 ``[1, n_mels, frames]`` log-mel grid)."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
    ) -> None:
        require_dependency("torchaudio", "MelSpectrogram")
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)

    def params(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
        }

    @staticmethod
    def kernel(value: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(params["sample_rate"]),
            n_fft=int(params["n_fft"]),
            hop_length=int(params["hop_length"]),
            n_mels=int(params["n_mels"]),
        )(value)
        return torch.log(mel + 1e-6), {}
