"""Cox survival and quantile forecast objectives: the loss values and the activation outputs."""

from __future__ import annotations

import unittest

import torch

from teia.node.net.activation.cox import CoxActivation
from teia.node.net.activation.quantile import QuantileActivation
from teia.node.net.loss.quantile import QuantileLoss
from teia.node.net.loss.survival_cox import CoxLoss


def breslow(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative partial log-likelihood with every tied time in the same risk set, written as the formula."""
    total = torch.zeros(())
    for i in range(len(risk)):
        if event[i]:
            total = total + risk[i] - torch.logsumexp(risk[time >= time[i]], dim=0)
    return -total / event.sum()


class CoxLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.risk = torch.randn(8)
        self.event = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])

    def test_matches_the_formula_without_ties(self) -> None:
        time = torch.tensor([5.0, 3, 8, 1, 9, 2, 7, 4])
        self.assertTrue(torch.allclose(CoxLoss()(self.risk, time, self.event), breslow(self.risk, time, self.event), atol=1e-6))

    def test_matches_the_formula_with_ties(self) -> None:
        time = torch.tensor([5.0, 5, 8, 1, 8, 2, 5, 4])
        self.assertTrue(torch.allclose(CoxLoss()(self.risk, time, self.event), breslow(self.risk, time, self.event), atol=1e-6))

    def test_does_not_depend_on_batch_order(self) -> None:
        time = torch.tensor([5.0, 5, 8, 1, 8, 2, 5, 4])
        perm = torch.randperm(len(time))
        shuffled = CoxLoss()(self.risk[perm], time[perm], self.event[perm])
        self.assertTrue(torch.allclose(CoxLoss()(self.risk, time, self.event), shuffled, atol=1e-6))

    def test_accepts_column_inputs_and_is_finite_without_events(self) -> None:
        time = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
        loss = CoxLoss()(self.risk.reshape(-1, 1), time, torch.zeros(8, 1))
        self.assertTrue(torch.isfinite(loss))

    def test_gradient_flows_to_the_risk(self) -> None:
        risk = self.risk.clone().requires_grad_()
        CoxLoss()(risk, torch.tensor([5.0, 5, 8, 1, 8, 2, 5, 4]), self.event).backward()
        self.assertTrue(torch.isfinite(risk.grad).all() and risk.grad.abs().sum() > 0)


class CoxActivationTests(unittest.TestCase):
    def test_risk_is_a_flat_vector_and_the_kernel_lists_floats(self) -> None:
        out = CoxActivation().activation(torch.tensor([[0.5], [-1.0]]))
        self.assertEqual(out["risk"].shape, (2,))
        self.assertEqual(CoxActivation.kernel(out, {}), [{"risk": 0.5}, {"risk": -1.0}])


class QuantileLossTests(unittest.TestCase):
    def test_pinball_values(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([2.5])
        # errors 1.5, 0.5, -0.5 at levels 0.1, 0.5, 0.9 -> 0.15, 0.25, 0.05
        self.assertAlmostEqual(float(QuantileLoss()(logits, target)), 0.15, places=6)

    def test_target_without_the_quantile_axis_is_aligned(self) -> None:
        logits = torch.zeros(4, 6, 3)
        target = torch.ones(4, 6)
        expected = (0.1 + 0.5 + 0.9) / 3
        self.assertAlmostEqual(float(QuantileLoss()(logits, target)), expected, places=6)

    def test_minimum_is_the_empirical_quantile(self) -> None:
        target = torch.arange(101, dtype=torch.float32)
        loss = QuantileLoss(quantiles=(0.9,))
        levels = torch.linspace(0, 100, 101)
        values = torch.stack([loss(torch.full((101, 1), float(v)), target) for v in levels])
        self.assertEqual(float(levels[values.argmin()]), 90.0)


class QuantileActivationTests(unittest.TestCase):
    def test_quantiles_never_cross(self) -> None:
        out = QuantileActivation().activation(torch.tensor([[[3.0, 1.0, 2.0]]]))["quantiles"]
        self.assertEqual(out.flatten().tolist(), [1.0, 2.0, 3.0])

    def test_levels_are_published(self) -> None:
        self.assertEqual(QuantileActivation(quantiles=(0.25, 0.75)).params(), {"levels": [0.25, 0.75]})

    def test_kernel_returns_one_row_per_sample(self) -> None:
        rows = QuantileActivation.kernel({"quantiles": torch.zeros(2, 4, 3)}, {})
        self.assertEqual([r["quantiles"].shape for r in rows], [(4, 3), (4, 3)])


if __name__ == "__main__":
    unittest.main()
