"""Off-policy discrete control task contract.

Source: common knowledge

Description:
  Replay transitions in; a discrete action index out, fed back as `action`. See teia:base/task.md.
"""

from __future__ import annotations

from teia.base.task import Dim, TaskContract, Tensor
from teia.task._rl_fields import REPLAY, TRANSITION


class RlDiscrete(TaskContract):
    """Off-policy discrete task contract."""

    meta = {"obs_state_dim": Dim(), "num_actions": Dim()}
    batch = {**TRANSITION, **REPLAY}
    capture = {"policy.action": Tensor("B *")}
    engine_atom = "policy.action"
    feedback = {"policy.action": "action"}
