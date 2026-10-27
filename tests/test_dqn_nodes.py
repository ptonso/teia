"""DQN temporal-difference loss and epsilon-greedy Q activation against their textbook definitions."""

import pytest
import torch

from teia.node.net.activation.q_argmax import ConstantSchedule, LinearSchedule, QArgmaxActivation
from teia.node.net.loss.dqn import DqnLoss

Q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ACTION = torch.tensor([1, 0])
REWARD = torch.tensor([1.0, 0.0])
Q_TARGET = torch.tensor([[0.0, 5.0], [2.0, 1.0]])
TERMINATED = torch.tensor([0.0, 1.0])
DISCOUNT = torch.tensor([0.9, 0.9])
# y = r + discount * (1 - done) * max_a' Q_target: [1 + 0.9 * 5, 0 + 0] = [5.5, 0]; Q(s, a) = [2, 3]


def test_squared_error_matches_the_bellman_target_by_hand() -> None:
    loss = DqnLoss(huber=False)(Q, ACTION, REWARD, Q_TARGET, TERMINATED, DISCOUNT)
    assert loss.item() == pytest.approx((0.5 * 3.5**2 + 0.5 * 3.0**2) / 2)


def test_huber_is_linear_beyond_one() -> None:
    loss = DqnLoss(huber=True)(Q, ACTION, REWARD, Q_TARGET, TERMINATED, DISCOUNT)
    assert loss.item() == pytest.approx(((3.5 - 0.5) + (3.0 - 0.5)) / 2)


def test_a_terminated_transition_does_not_bootstrap() -> None:
    q_target = torch.tensor([[0.0, 5.0], [100.0, 100.0]])
    loss = DqnLoss(huber=False)(Q, ACTION, REWARD, q_target, TERMINATED, DISCOUNT)
    assert loss.item() == pytest.approx((0.5 * 3.5**2 + 0.5 * 3.0**2) / 2)


def test_gamma_applies_without_an_n_step_discount() -> None:
    loss = DqnLoss(gamma=0.5, huber=False)(Q, ACTION, REWARD, Q_TARGET, TERMINATED)
    assert loss.item() == pytest.approx((0.5 * (2.0 - 3.5) ** 2 + 0.5 * 3.0**2) / 2)


def test_sample_weights_scale_each_transition() -> None:
    loss = DqnLoss(huber=False)(Q, ACTION, REWARD, Q_TARGET, TERMINATED, DISCOUNT, weight=torch.tensor([2.0, 0.0]))
    assert loss.item() == pytest.approx(0.5 * 3.5**2 * 2 / 2)


def test_gradient_reaches_only_the_taken_action_and_never_the_target() -> None:
    q, q_target = Q.clone().requires_grad_(), Q_TARGET.clone().requires_grad_()
    DqnLoss()(q, ACTION, REWARD, q_target, TERMINATED, DISCOUNT).backward()
    assert q.grad[0, 0] == 0 and q.grad[1, 1] == 0 and q.grad[0, 1] != 0 and q.grad[1, 0] != 0
    assert q_target.grad is None


def test_the_deterministic_route_is_the_argmax() -> None:
    act = QArgmaxActivation(epsilon=1.0)
    assert act.activation(Q)["action"].tolist() == [1, 1]
    assert act.sample(Q, deterministic=True).action.tolist() == [1, 1]


def test_epsilon_zero_is_greedy_and_one_is_uniform() -> None:
    q = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4000, 1)
    assert QArgmaxActivation(epsilon=0.0).sample(q, deterministic=False).action.eq(1).all()
    counts = torch.bincount(QArgmaxActivation(epsilon=1.0).sample(q, deterministic=False, generator=torch.Generator().manual_seed(0)).action, minlength=3)
    assert (counts / 4000).tolist() == pytest.approx([1 / 3] * 3, abs=0.05)


def test_epsilon_decays_linearly_in_environment_steps() -> None:
    act = QArgmaxActivation(epsilon={"start": 1.0, "end": 0.0, "duration": 100})
    assert act.epsilon == 1.0
    act.sample(torch.zeros(25, 2), deterministic=False)
    assert act.epsilon == pytest.approx(0.75)
    act.sample(torch.zeros(75, 2), deterministic=False)
    assert act.epsilon == 0.0


def test_evaluation_does_not_advance_the_schedule() -> None:
    act = QArgmaxActivation(epsilon={"start": 1.0, "end": 0.0, "duration": 100})
    act.sample(torch.zeros(50, 2), deterministic=True)
    assert act.epsilon == 1.0


def test_schedules_accept_scalars_and_instances() -> None:
    assert QArgmaxActivation(epsilon=0.3).epsilon == 0.3
    assert QArgmaxActivation(epsilon=LinearSchedule(start=0.5, end=0.5)).epsilon == 0.5
    assert QArgmaxActivation(epsilon={"_target_": "x.ConstantSchedule", "value": 0.2}).epsilon == ConstantSchedule(value=0.2).value(0)
    with pytest.raises(TypeError):
        QArgmaxActivation(epsilon="high")
