import torch
import torch.nn.functional as F

from teia.node.net.loss.balanced_cross_entropy import BalancedCrossEntropyLoss


def batch(counts, classes=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    target = torch.cat([torch.full((n,), c) for c, n in enumerate(counts)])
    return torch.randn(len(target), classes, generator=generator), target


def test_equals_inverse_frequency_weighted_cross_entropy():
    logits, target = batch([40, 10, 400])
    counts = torch.bincount(target).float()
    weights = target.numel() / (len(counts) * counts)
    expected = F.cross_entropy(logits, target, weight=weights)
    assert torch.allclose(BalancedCrossEntropyLoss()(logits, target), expected, atol=1e-6)


def test_absent_class_is_skipped():
    logits, target = batch([30, 0, 200])
    present = [c for c in (0, 2)]
    expected = torch.stack([F.cross_entropy(logits[target == c], target[target == c]) for c in present]).mean()
    assert torch.allclose(BalancedCrossEntropyLoss()(logits, target), expected, atol=1e-6)


def test_single_class_batch_is_plain_cross_entropy():
    logits, target = batch([0, 0, 50])
    assert torch.allclose(BalancedCrossEntropyLoss()(logits, target), F.cross_entropy(logits, target), atol=1e-6)


def test_gradient_is_finite_with_label_smoothing():
    logits, target = batch([5, 1, 60])
    logits.requires_grad_(True)
    BalancedCrossEntropyLoss(label_smoothing=0.1)(logits, target).backward()
    assert torch.isfinite(logits.grad).all()
