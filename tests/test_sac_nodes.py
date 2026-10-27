"""Soft actor-critic losses and the tanh-squashed Gaussian against their textbook definitions."""

import math

import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from teia.node.net._gaussian_policy_utils import squashed_gaussian_sample
from teia.node.net.activation.tanh_gaussian_box import TanhGaussianBoxActivation
from teia.node.net.loss.sac_actor import SacActorLoss
from teia.node.net.loss.sac_alpha import SacAlphaLoss
from teia.node.net.loss.sac_critic import SacCriticLoss


def test_critic_regresses_both_critics_onto_the_soft_bellman_target() -> None:
    log_alpha = torch.tensor(math.log(0.5))
    loss = SacCriticLoss()(
        q1_buf=torch.tensor([[3.0], [1.0]]), q2_buf=torch.tensor([[1.0], [3.0]]), reward=torch.tensor([1.0, 2.0]),
        q1_targ=torch.tensor([[3.0], [4.0]]), q2_targ=torch.tensor([[2.0], [5.0]]), next_logp=torch.tensor([-1.0, 0.0]),
        terminated=torch.tensor([0.0, 1.0]), n_step_discount=torch.tensor([0.9, 0.9]), log_alpha=log_alpha,
    )
    # y = [1 + 0.9 (min(3, 2) + 0.5), 2] = [3.25, 2]
    assert loss.item() == pytest.approx(((3 - 3.25) ** 2 + (1 - 2) ** 2) / 2 + ((1 - 3.25) ** 2 + (3 - 2) ** 2) / 2)


def test_critic_target_carries_no_gradient() -> None:
    q1_targ, next_logp, log_alpha = torch.ones(2, 1, requires_grad=True), torch.zeros(2, requires_grad=True), torch.zeros((), requires_grad=True)
    q1_buf = torch.zeros(2, 1, requires_grad=True)
    SacCriticLoss()(q1_buf, q1_buf, torch.ones(2), q1_targ, q1_targ, next_logp, torch.zeros(2), torch.full((2,), 0.99), log_alpha).backward()
    assert q1_buf.grad is not None
    assert q1_targ.grad is None and next_logp.grad is None and log_alpha.grad is None


def test_actor_minimises_alpha_log_prob_minus_the_smaller_critic() -> None:
    log_alpha = torch.tensor(math.log(2.0), requires_grad=True)
    q1, q2, logp = (torch.tensor(v, requires_grad=True) for v in ([[1.0], [4.0]], [[2.0], [3.0]], [-1.0, 2.0]))
    loss = SacActorLoss()(q1, q2, logp, log_alpha)
    # alpha log pi = [-2, 4];  min q = [1, 3]
    assert loss.item() == pytest.approx(((-2 - 1) + (4 - 3)) / 2)
    loss.backward()
    assert log_alpha.grad is None and logp.grad is not None and q1.grad[0] != 0 and q2.grad[0] == 0


def test_temperature_loss_pushes_alpha_toward_the_target_entropy() -> None:
    log_alpha, logp = torch.tensor(0.5, requires_grad=True), torch.tensor([-2.0, 0.0], requires_grad=True)
    loss = SacAlphaLoss(target_entropy=-1.0)(logp, log_alpha)
    # -log_alpha * mean(log pi + H) = -0.5 * ((-3 + -1) / 2)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert log_alpha.grad.item() == pytest.approx(2.0) and logp.grad is None


def test_temperature_target_entropy_is_minus_the_action_dimension() -> None:
    class Dm:
        action_dim = 3

    loss = SacAlphaLoss()
    loss.configure_from_datamodule(Dm())
    assert loss.target_entropy == -3.0


def test_temperature_without_a_target_fails() -> None:
    with pytest.raises(ValueError, match="target_entropy"):
        SacAlphaLoss()(torch.zeros(2), torch.zeros(()))
    with pytest.raises(AttributeError):
        SacAlphaLoss().configure_from_datamodule(object())


def test_squashed_log_prob_follows_the_change_of_variables() -> None:
    mean, log_std = torch.tensor([[0.3, -0.4]]).repeat(64, 1), torch.tensor([-0.5, 0.2])
    action, log_prob = squashed_gaussian_sample(mean, log_std, generator=torch.Generator().manual_seed(0))
    reference = TransformedDistribution(Normal(mean, log_std.exp().expand_as(mean)), [TanhTransform()])
    assert torch.allclose(log_prob, reference.log_prob(action.clamp(-0.99999, 0.99999)).sum(-1), atol=1e-3)


def test_squashed_action_is_mapped_into_the_bounds() -> None:
    mean, log_std = torch.randn(512, 2) * 3, torch.zeros(2)
    action, _ = squashed_gaussian_sample(mean, log_std, low=-2.0, high=[2.0, 4.0])
    assert action[:, 0].abs().max() <= 2.0 and action[:, 1].min() >= -2.0 and action[:, 1].max() <= 4.0


def test_deterministic_squashed_action_is_tanh_of_the_mean() -> None:
    action, _ = squashed_gaussian_sample(torch.tensor([[0.7]]), torch.zeros(1), deterministic=True, low=-2.0, high=2.0)
    assert action.item() == pytest.approx(2.0 * math.tanh(0.7))


def test_tanh_gaussian_activation_samples_inside_the_bounds_and_reports_entropy() -> None:
    act = TanhGaussianBoxActivation(low=-2.0, high=2.0)
    sample = act.sample(torch.randn(256, 1), torch.zeros(1), deterministic=False)
    assert sample.action.abs().max() <= 2.0
    assert torch.equal(sample.entropy, -sample.log_prob)
    assert act.activation(torch.tensor([[0.5]]), torch.zeros(1))["action"].item() == pytest.approx(2.0 * math.tanh(0.5))
