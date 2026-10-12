"""
InceptionPool3: the standard FID feature extractor.

Source:
  - title: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    url: "https://arxiv.org/abs/1706.08500"
    year: 2017
  - title: "pytorch-fid"
    url: "https://github.com/mseitzer/pytorch-fid"
    year: 2020

Description:
  Pool3 (2048-d) activations of the TensorFlow FID Inception-V3 as ported by pytorch-fid, the
  network behind the FID numbers most papers report. Input is ``(N, 3, H, W)`` in ``[0, 1]``; the
  network resizes bilinearly to 299 and rescales to ``[-1, 1]`` itself. The weights are downloaded
  once into the torch hub cache and their full sha256 is verified on every construction.

Upstream:
  package: pytorch-fid==0.3.0
  weights: https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3
from torch import nn

__all__ = ["InceptionPool3", "ensure_weights"]

SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"


def ensure_weights() -> Path:
    """Download the FID weights into the torch hub cache when absent, then verify their sha256."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / FID_WEIGHTS_URL.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(FID_WEIGHTS_URL, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"FID weights at {path} (from {FID_WEIGHTS_URL}) have sha256 {digest}, expected {SHA256}.")
    return path


class InceptionPool3(nn.Module):
    """FID Inception-V3 pool3 features (Heusel et al. 2017): ``(N, 3, H, W)`` in ``[0, 1]`` to ``(N, 2048)``."""

    identity = f"pytorch-fid-0.3.0/inception-v3-pool3/sha256:{SHA256}"

    def __init__(self) -> None:
        super().__init__()
        ensure_weights()
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True)
        self.eval()

    def train(self, mode: bool = True) -> "InceptionPool3":
        """Always eval: a train-mode call would update BatchNorm statistics and change the features."""
        return super().train(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)[0].flatten(1)
