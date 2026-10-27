"""Off-policy continuous control task contract.

Source: common knowledge

Description:
  Replay transitions (next state, n-step discount) in; a continuous action out, fed back as `action`. See teia:base/task.md.
"""

from __future__ import annotations

from teia.base.task import Dim, TaskContract, Tensor
from teia.task._rl_fields import REPLAY, TRANSITION


class RlContinuousReplay(TaskContract):
    """Off-policy continuous task contract."""

    meta = {"obs_state_dim": Dim(), "action_dim": Dim()}
    batch = {**TRANSITION, **REPLAY}
    capture = {"policy.action": Tensor("B *")}
    engine_atom = "policy.action"
    feedback = {"policy.action": "action"}
