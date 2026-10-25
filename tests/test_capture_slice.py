"""``TeiaNetModule.act_forward`` runs only the nodes the ``capture:`` map needs (teia:core/module/capture_map.md#capture-slice)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from teia.base.net import TeiaNode
from teia.core.module.net import TeiaNetModule

CALLS: list[str] = []


class Head(TeiaNode):
    def build_module(self, label: str = "head") -> None:
        self.label = label
        self.linear = nn.Linear(self.in_shape[0], self.out_shape[0])

    def forward(self, x):
        CALLS.append(self.label)
        return self.linear(x)


def _head(label, key, out_key, size):
    return {
        "_target_": "test_capture_slice.Head",
        "label": label,
        "in": key,
        "out": {out_key: (size,)},
    }


def _module():
    module = TeiaNetModule(
        encoder=_head("encoder", "batch.obs", "feat.x", 8),
        policy_head=_head("policy", "feat.x", "pred.policy", 3),
        value_head=_head("value", "feat.x", "pred.value", 1),
        critic=_head("critic", "feat.x", "pred.q", 3),
        act={"_target_": "teia.node.net.activation.categorical.CategoricalActivation", "in": "pred.policy", "out": {"act.action": None, "act.log_prob": None}},
        capture={"policy.action": "act.action", "policy.log_prob": "act.log_prob", "policy.value": "pred.value"},
    )
    dm = SimpleNamespace(batch_meta=lambda: {"obs": (4,)}, meta=lambda: {})
    module.configure_from_datamodule(dm)
    return module


def test_acting_runs_only_the_nodes_the_capture_map_needs():
    module = _module()
    CALLS.clear()
    out = module.act_forward(SimpleNamespace(obs=torch.randn(5, 4)), deterministic=True)
    assert sorted(CALLS) == ["encoder", "policy", "value"]
    assert set(out) == {"policy.action", "policy.log_prob", "policy.value"}
    assert out["policy.action"].shape == (5,) and out["policy.value"].shape == (5, 1)


def test_the_full_forward_still_runs_every_node():
    module = _module()
    CALLS.clear()
    module(SimpleNamespace(obs=torch.randn(2, 4)))
    assert sorted(CALLS) == ["critic", "encoder", "policy", "value"]


def test_acting_carries_no_graph_and_samples_unless_deterministic():
    module = _module()
    batch = SimpleNamespace(obs=torch.randn(64, 4))
    out = module.act_forward(batch)
    assert not out["policy.value"].requires_grad
    greedy = module.act_forward(batch, deterministic=True)["policy.action"]
    assert torch.equal(greedy, module.act_forward(batch, deterministic=True)["policy.action"])


def test_a_post_source_cannot_serve_acting():
    module = _module()
    module._capture_spec = {"policy.action": "post.act.label"}
    with pytest.raises(ValueError, match="post"):
        module.act_forward(SimpleNamespace(obs=torch.randn(2, 4)))
