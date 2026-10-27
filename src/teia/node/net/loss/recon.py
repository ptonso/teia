"""
Reconstruction Loss.

Source:
  - title: "Auto-Encoding Variational Bayes"
    url: "https://arxiv.org/abs/1312.6114"
    year: 2013
  - title: "Simple and Effective VAE Training with Calibrated Decoders"
    url: "https://arxiv.org/abs/2006.13202"
    year: 2021

Description:
  Pixel reconstruction error between decoder logits, mapped into the pixel space of the task, and the target image. The ``sigma`` option reads the L2 error as the negative log-likelihood of a Gaussian decoder. ``fixed`` keeps the plain summed error. ``optimal`` takes the maximum-likelihood variance, the mean squared error, as one shared value for the batch at every step, and ``optimal_per_image`` gives each image its own, so the per-sample NLL over D pixels is D/2 (log MSE + 1), with the constant D/2 log 2 pi dropped. Pair the calibrated modes with a KL weight of 1.

Adaptations:
  - ``strategy: none`` skips the squash, so the loss also fits a target that already lives in the output space, such as a latent.
  - The paper soft-clips log sigma at -6; the floor is left out, since the MSE of images in [0, 1] stays far above e^-12.
"""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss
from teia.node.net._pixel_space import squash


class ReconLoss(BaseLoss):
    """Pixel reconstruction loss, mean over the batch of a per-sample sum over D pixels.

        fixed:              L = mean_b sum_pixels l(x_hat - x)
        optimal:            L = D/2 (log MSE_batch + 1)
        optimal_per_image:  L = mean_b D/2 (log MSE_b + 1)

    Args:
        recon_type: ``l2``, ``l1`` or ``smooth_l1`` for ``l`` (the calibrated modes need ``l2``).
        huber_delta: ``beta`` of ``smooth_l1``.
        strategy: Pixel-norm space of the target: ``unit`` or ``signed``, or ``none`` for an unsquashed target such as a latent.
        sigma: Decoder variance: ``fixed`` (plain summed error), ``optimal`` (shared) or ``optimal_per_image``.
    """

    def __init__(
        self,
        recon_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        huber_delta: float = 1.0,
        strategy: str = "unit",
        sigma: Literal["fixed", "optimal", "optimal_per_image"] = "fixed",
    ) -> None:
        super().__init__()
        if sigma not in ("fixed", "optimal", "optimal_per_image"):
            raise ValueError(f"unknown sigma {sigma!r}")
        if sigma != "fixed" and recon_type not in ("l2", "mse"):
            raise ValueError(f"sigma={sigma!r} is a Gaussian likelihood and needs recon_type l2, got {recon_type!r}")
        self.recon_type = recon_type
        self.huber_delta = float(huber_delta)
        self.strategy = strategy
        self.sigma = sigma

    def forward(self, reconstructed: Tensor, img: Tensor) -> Tensor:
        if self.strategy != "none":
            reconstructed = squash(reconstructed, self.strategy)
        if self.recon_type in ("l2", "mse"):
            per = F.mse_loss(reconstructed, img, reduction="none")
        elif self.recon_type in ("l1", "mae"):
            per = F.l1_loss(reconstructed, img, reduction="none")
        else:
            per = F.smooth_l1_loss(reconstructed, img, reduction="none", beta=self.huber_delta)
        per = per.flatten(1)
        half_d = per.shape[1] / 2
        if self.sigma == "optimal":
            return half_d * (per.mean().log() + 1)
        if self.sigma == "optimal_per_image":
            return (half_d * (per.mean(1).log() + 1)).mean()
        return per.sum(1).mean()


__all__ = ["ReconLoss"]
