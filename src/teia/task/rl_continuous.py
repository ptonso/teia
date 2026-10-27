"""On-policy continuous control task contract.

Source: common knowledge

Description:
  Rollout transitions with log-prob, value, advantage and return in; a continuous action, its log-probability and the value out, all fed back. See teia:base/task.md.
"""

from __future__ import annotations

from teia.base.task import Dim, TaskContract, Tensor
from teia.task._rl_fields import ON_POLICY, TRANSITION


class RlContinuous(TaskContract):
    """On-policy continuous task contract."""

    meta = {"obs_state_dim": Dim(), "action_dim": Dim()}
    batch = {**TRANSITION, **ON_POLICY}
    capture = {"policy.action": Tensor("B *"), "policy.env_action": Tensor("B *"), "policy.log_prob": Tensor("B"), "policy.value": Tensor("B")}
    engine_atom = "policy.env_action"
    feedback = {"policy.action": "action", "policy.log_prob": "log_prob", "policy.value": "value"}
