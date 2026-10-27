"""Gaussian policy activation, clipped PPO surrogate and value regression against their textbook definitions."""

import math

import pytest
import torch

from teia.node.net.activation.gaussian_box import GaussianBoxActivation
from teia.node.net.loss.ppo_policy import PpoPolicyLoss
from teia.node.net.loss.value import ValueLoss

STANDARD_NORMAL_LOGP = {0.0: -0.5 * math.log(2 * math.pi), 1.0: -0.5 * math.log(2 * math.pi) - 0.5}


def _logp(action: float) -> float:
    return STANDARD_NORMAL_LOGP[action]


def test_ppo_surrogate_clips_the_ratio_on_the_side_that_helps() -> None:
    mean, log_std, action = torch.zeros(2, 1), torch.zeros(1), torch.tensor([[0.0], [1.0]])
    old = torch.tensor([_logp(0.0) - math.log(1.5), _logp(1.0) + math.log(2.0)])
    loss = PpoPolicyLoss()(mean, log_std, action, old, torch.tensor([1.0, -1.0]))
    # ratio 1.5 with A=1 -> -min(1.5, 1.2) = -1.2;  ratio 0.5 with A=-1 -> max(0.5, 0.8) = 0.8
    assert loss.item() == pytest.approx((-1.2 + 0.8) / 2)


def test_ppo_surrogate_at_ratio_one_is_minus_the_mean_advantage() -> None:
    mean, log_std, action = torch.zeros(2, 1), torch.zeros(1), torch.tensor([[0.0], [1.0]])
    old = torch.tensor([_logp(0.0), _logp(1.0)])
    assert PpoPolicyLoss()(mean, log_std, action, old, torch.tensor([1.0, 3.0])).item() == pytest.approx(-2.0)


def test_ppo_advantage_normalisation_standardises_the_batch() -> None:
    mean, log_std, action = torch.zeros(2, 1), torch.zeros(1), torch.tensor([[0.0], [1.0]])
    old = torch.tensor([_logp(0.0), _logp(1.0)])
    loss = PpoPolicyLoss(normalize_advantage=True)(mean, log_std, action, old, torch.tensor([1.0, 3.0]))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_ppo_gradient_reaches_the_mean_and_the_log_std() -> None:
    mean, log_std = torch.zeros(2, 1, requires_grad=True), torch.zeros(1, requires_grad=True)
    PpoPolicyLoss()(mean, log_std, torch.tensor([[0.5], [-0.5]]), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])).backward()
    assert mean.grad.abs().sum() > 0 and log_std.grad.abs().sum() > 0


def test_value_loss_is_half_the_mean_squared_error() -> None:
    assert ValueLoss()(torch.tensor([[1.0], [3.0]]), torch.tensor([[0.0], [1.0]])).item() == pytest.approx(0.5 * (1 + 4) / 2)


def test_clipped_value_loss_takes_the_larger_of_the_two_errors() -> None:
    loss = ValueLoss(clip_coef=0.5)(torch.tensor([2.5]), torch.tensor([2.4]), torch.tensor([1.0]))
    # unclipped (0.1)^2 = 0.01;  clipped value 1 + 0.5 = 1.5 gives (0.9)^2 = 0.81
    assert loss.item() == pytest.approx(0.5 * 0.81)


def test_gaussian_box_deterministic_route_is_the_mean_clamped_to_the_bounds() -> None:
    act = GaussianBoxActivation(low=-1.0, high=1.0)
    out = act.activation(torch.tensor([[0.5, 3.0]]), torch.zeros(2))
    assert out["action"].tolist() == [[0.5, 3.0]] and out["env_action"].tolist() == [[0.5, 1.0]]


def test_gaussian_box_sample_keeps_the_raw_action_for_the_log_probability() -> None:
    act = GaussianBoxActivation(low=-1.0, high=1.0)
    mean, log_std = torch.full((4096, 1), 2.0), torch.zeros(1)
    sample = act.sample(mean, log_std, deterministic=False, generator=torch.Generator().manual_seed(0))
    assert sample.action.mean().item() == pytest.approx(2.0, abs=0.1) and sample.action.std().item() == pytest.approx(1.0, abs=0.1)
    assert sample.env_action.max().item() == 1.0
    expected = torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(sample.action).sum(-1)
    assert torch.allclose(sample.log_prob, expected)
    assert sample.entropy.mean().item() == pytest.approx(0.5 * math.log(2 * math.pi * math.e))


def test_gaussian_box_deterministic_sample_is_the_mean() -> None:
    sample = GaussianBoxActivation().sample(torch.tensor([[0.3]]), torch.zeros(1), deterministic=True)
    assert sample.action.item() == pytest.approx(0.3)
