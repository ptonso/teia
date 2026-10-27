"""GymnasiumEngine: signals, partial reset, spaces, seeds and the action bound check (teia:base/data/nodes.md#engine)."""

import pytest
import torch

pytest.importorskip("gymnasium")

from teia.node.data.engine.gymnasium import GymnasiumEngine

STATE = {"item.obs": "observation", "item.reward": "reward", "item.terminated": "terminated", "item.truncated": "truncated"}


def _named(engine: GymnasiumEngine, world: str) -> GymnasiumEngine:
    engine.out_key, engine.signals, engine.data_root = list(STATE), list(STATE.values()), world
    return engine


def test_a_finished_environment_keeps_its_final_observation_until_reset() -> None:
    engine = _named(GymnasiumEngine(num_envs=3, seed=1), "gym:CartPole-v1")
    engine.reset()
    for _ in range(80):
        obs, _, terminated, truncated = engine.step(torch.zeros(3, dtype=torch.int64))
        ended = terminated | truncated
        if ended.any():
            i = int(ended.nonzero()[0])
            assert obs[i].abs().max() > 0.2
            fresh, *_ = engine.reset(ended)
            assert fresh[i].abs().max() < 0.06
            return
    pytest.fail("no episode ended")


def test_seeds_are_applied() -> None:
    first = _named(GymnasiumEngine(num_envs=2, seed=3), "gym:CartPole-v1").reset()[0]
    again = _named(GymnasiumEngine(num_envs=2, seed=3), "gym:CartPole-v1").reset()[0]
    other = _named(GymnasiumEngine(num_envs=2, seed=4), "gym:CartPole-v1").reset()[0]
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_the_network_owns_the_range_so_an_out_of_range_action_fails() -> None:
    engine = _named(GymnasiumEngine(num_envs=2), "gym:Pendulum-v1")
    engine.reset()
    engine.step(torch.tensor([[2.0], [-2.0]]))
    with pytest.raises(ValueError, match="outside the simulator's bounds"):
        engine.step(torch.tensor([[3.0], [0.0]]))


def test_spec_reports_roles_actions_and_dims() -> None:
    engine = _named(GymnasiumEngine(num_envs=2), "gym:CartPole-v1")
    spec = engine.spec()
    assert [role for *_, role in spec.outputs] == ["state", "outcome", "outcome", "outcome"]
    assert (spec.num_actions, spec.action_low) == (2, None)
    assert engine.runtime_dims()["obs_dim"] == 4


def test_the_world_must_be_a_gym_id() -> None:
    with pytest.raises(ValueError, match="gym:<id>"):
        _named(GymnasiumEngine(num_envs=1), "/some/folder").open(0)


def test_an_unknown_signal_fails_at_open() -> None:
    engine = _named(GymnasiumEngine(num_envs=1), "gym:CartPole-v1")
    engine.signals = ["nonsense"]
    with pytest.raises(ValueError, match="no signal"):
        engine.open(0)
