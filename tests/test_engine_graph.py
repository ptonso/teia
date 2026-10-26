import pytest

from teia.core.datamodule.graph import build_graph
from teia.base.data import Engine, Transform

from _fake_engine import FakeEngine

_ENGINE = "_fake_engine.FakeEngine"
OUT = {"item.obs": "obs", "item.reward": "reward", "item.terminated": "terminated", "item.truncated": "truncated"}


class Passthrough(Transform):
    def __call__(self, value):
        return value


def test_an_engine_is_a_stream_source_and_maps_its_outputs_to_signals():
    records = build_graph([("env", {"_target_": _ENGINE, "out": OUT})])
    (env,) = records
    assert (env.kind, env.phase, env.stage) == ("engine", "source", "stream")
    assert env.in_key == [] and env.out_key[0] == "item.obs" and env.signals[0] == "obs"


def test_an_engine_takes_no_input():
    with pytest.raises(ValueError, match="output-only"):
        build_graph([("env", {"_target_": _ENGINE, "in": "item.x", "out": {"item.obs": "obs"}})])


def test_a_mapping_in_binds_roles_and_orders_the_dependencies():
    records = build_graph(
        [
            ("env", {"_target_": _ENGINE, "out": OUT}),
            ("norm", {"_target_": "test_engine_graph.Passthrough", "in": {"value": "item.obs"}, "out": "item.obs_n"}),
        ]
    )
    norm = next(r for r in records if r.name == "norm")
    assert norm.in_roles == {"value": "item.obs"} and norm.in_key == ["item.obs"]


def test_engine_dims_follow_the_out_keys_and_the_action_spec():
    engine = FakeEngine()
    engine.out_key = ["item.obs_state", "item.reward", "item.terminated", "item.truncated"]
    engine.signals = ["obs", "reward", "terminated", "truncated"]
    assert engine.runtime_dims() == {
        "obs_state_shape": (2,),
        "obs_state_dim": 2,
        "reward_shape": (),
        "terminated_shape": (),
        "truncated_shape": (),
        "action_dim": 1,
    }
    assert isinstance(engine, Engine)
