import pytest
import torch

from teia.core.datamodule.interactive.rollout import RolloutBuffer


ROLES = {"reward": "item.r", "terminated": "item.term", "truncated": "item.trunc", "value": "item.v", "valid": "item.ok"}
OUT = ["item.ret", "item.adv"]


def _buffer(**kwargs):
    buffer = RolloutBuffer(**kwargs)
    buffer.in_roles, buffer.out_key = dict(ROLES), list(OUT)
    return buffer


def _advantages(*, terminated_at=None, truncated_at=None, mask_at=None):
    buffer = _buffer(rollout_steps=3, minibatch_size=8, update_epochs=1, gamma=0.5, gae_lambda=1.0, normalize_advantage=False)
    buffer.configure(num_envs=1, generator=torch.Generator().manual_seed(0))
    for t in range(3):
        buffer.add(
            {
                "item.t": torch.tensor([float(t)]),
                "item.r": torch.ones(1),
                "item.v": torch.full((1,), 0.5),
                "item.term": torch.tensor([t == terminated_at]),
                "item.trunc": torch.tensor([t == truncated_at]),
                "item.ok": torch.tensor([0.0 if t == mask_at else 1.0]),
            }
        )
    buffer.finalize_rollout(last_value=torch.ones(1))
    samples = next(buffer.sample_iter())
    by_step = {int(sample["item.t"]): sample for sample in samples}
    return {t: (float(s["item.adv"]), float(s["item.ret"])) for t, s in by_step.items()}


def test_gae_matches_the_hand_computed_trace() -> None:
    result = _advantages()
    assert result[2] == pytest.approx((1.0, 1.5))
    assert result[1] == pytest.approx((1.25, 1.75))
    assert result[0] == pytest.approx((1.375, 1.875))


def test_termination_zeroes_the_bootstrap_and_cuts_the_trace() -> None:
    result = _advantages(terminated_at=1)
    assert result[1][0] == pytest.approx(0.5)
    assert result[0][0] == pytest.approx(1.0)


def test_truncation_bootstraps_but_cuts_the_trace() -> None:
    result = _advantages(truncated_at=1)
    assert result[1][0] == pytest.approx(0.75)
    assert result[0][0] == pytest.approx(1.125)


def test_invalid_rows_are_dropped_and_advantages_normalize_over_the_rest() -> None:
    buffer = _buffer(rollout_steps=4, minibatch_size=8, update_epochs=1)
    buffer.configure(num_envs=2)
    for t in range(4):
        buffer.add(
            {
                "item.r": torch.full((2,), float(t)),
                "item.v": torch.zeros(2),
                "item.term": torch.zeros(2, dtype=torch.bool),
                "item.trunc": torch.zeros(2, dtype=torch.bool),
                "item.ok": torch.tensor([1.0, 0.0 if t == 2 else 1.0]),
            }
        )
    buffer.finalize_rollout(last_value=torch.zeros(2))
    samples = next(buffer.sample_iter())
    advantages = torch.stack([sample["item.adv"] for sample in samples])
    assert len(samples) == 7
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(advantages.std()) == pytest.approx(1.0, abs=1e-3)


def test_a_buffer_refuses_unbound_or_unknown_roles() -> None:
    buffer = RolloutBuffer()
    buffer.in_roles, buffer.out_key = {"reward": "item.r"}, list(OUT)
    with pytest.raises(ValueError, match="missing"):
        buffer.configure(num_envs=1)
    buffer.in_roles = {**ROLES, "extra": "item.x"}
    with pytest.raises(ValueError, match="unknown"):
        buffer.configure(num_envs=1)
