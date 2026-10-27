"""Reconstruction objective and activation, the pixel-space helpers they share, and the FID device choice."""

from __future__ import annotations

import math

import pytest
import torch

from teia.core.export.collect import collect_callable
from teia.node.eval.metric.unpaired._embedding import EmbeddingAccumulator
from teia.node.net._pixel_space import squash, to_strategy, to_unit
from teia.node.net.activation.reconstruct import ReconstructActivation
from teia.node.net.loss.recon import ReconLoss


def _batch():
    torch.manual_seed(0)
    return torch.randn(4, 3, 8, 8, requires_grad=True), torch.rand(4, 3, 8, 8)


def test_fixed_is_the_summed_squared_error_per_sample():
    logits, img = _batch()
    expected = (torch.sigmoid(logits) - img).pow(2).flatten(1).sum(1).mean()
    assert torch.allclose(ReconLoss()(logits, img), expected)


def test_l1_and_smooth_l1_sum_per_sample():
    logits, img = _batch()
    diff = torch.sigmoid(logits) - img
    assert torch.allclose(ReconLoss(recon_type="l1")(logits, img), diff.abs().flatten(1).sum(1).mean())
    small = diff.abs() < 1
    smooth = torch.where(small, 0.5 * diff.pow(2), diff.abs() - 0.5)
    assert torch.allclose(ReconLoss(recon_type="smooth_l1")(logits, img), smooth.flatten(1).sum(1).mean())


def test_signed_strategy_squashes_with_tanh_and_none_skips_the_squash():
    logits, img = _batch()
    assert torch.allclose(ReconLoss(strategy="signed")(logits, img), (torch.tanh(logits) - img).pow(2).flatten(1).sum(1).mean())
    assert torch.allclose(ReconLoss(strategy="none")(logits, img), (logits - img).pow(2).flatten(1).sum(1).mean())


def test_optimal_is_gaussian_nll_at_the_ml_sigma():
    logits, img = _batch()
    per = (torch.sigmoid(logits) - img).pow(2).flatten(1)
    d = per.shape[1]
    var = per.mean()
    nll = (per.sum(1) / (2 * var) + d / 2 * var.log()).mean()
    assert torch.allclose(ReconLoss(sigma="optimal")(logits, img), nll)
    var_b = per.mean(1)
    nll_b = (per.sum(1) / (2 * var_b) + d / 2 * var_b.log()).mean()
    assert torch.allclose(ReconLoss(sigma="optimal_per_image")(logits, img), nll_b)
    assert math.isclose(nll_b.item(), (d / 2 * (var_b.log() + 1)).mean().item(), rel_tol=1e-5)


def test_calibrated_modes_have_finite_gradients():
    for sigma in ("optimal", "optimal_per_image"):
        logits, img = _batch()
        ReconLoss(sigma=sigma)(logits, img).backward()
        assert torch.isfinite(logits.grad).all()


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="needs recon_type l2"):
        ReconLoss(recon_type="l1", sigma="optimal")
    with pytest.raises(ValueError, match="unknown sigma"):
        ReconLoss(sigma="learned")
    with pytest.raises(ValueError, match="unknown pixel_norm"):
        ReconLoss(strategy="imagenet")(*_batch())


def test_activation_squashes_into_the_pixel_space_and_exports_unit_images():
    logits = torch.tensor([[[[-20.0, 0.0, 20.0]]]])
    unit = ReconstructActivation("unit")
    signed = ReconstructActivation("signed")
    assert torch.allclose(unit.activation(logits)["reconstructed"], torch.sigmoid(logits))
    assert torch.allclose(signed.activation(logits)["reconstructed"], torch.tanh(logits))
    assert signed.params() == {"strategy": "signed"}
    image = ReconstructActivation.kernel(signed.activation(logits), signed.params())[0]["image"]
    assert image.min() >= 0 and image.max() <= 1 and torch.allclose(image[0, 0, 1], torch.tensor(0.5))


def test_pixel_space_round_trip():
    x = torch.linspace(-1, 1, 9)
    assert torch.allclose(to_strategy(to_unit(x, "signed"), "signed"), x)
    assert torch.equal(squash(x, "unit"), torch.sigmoid(x))
    with pytest.raises(ValueError):
        to_unit(x, "none")


def test_fid_device_auto_follows_cuda_availability():
    accumulator = EmbeddingAccumulator(torch.nn.Identity(), device="auto")
    assert accumulator.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert EmbeddingAccumulator(torch.nn.Identity(), device="cpu").device.type == "cpu"


def test_activation_kernel_is_export_collectable():
    emitted = collect_callable(ReconstructActivation.kernel, owner="ReconstructActivation")
    assert emitted.entry == "reconstructactivation_kernel" and "teia" not in emitted.render()
