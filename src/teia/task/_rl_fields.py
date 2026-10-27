"""Field sets the RL task contracts share (teia:base/task.md#field-registration).

Source: common knowledge

Description:
  The transition fields every RL contract declares, with the on-policy and replay additions. Reward, termination, truncation and episode id are eval targets.
"""

from __future__ import annotations

from teia.base.task import Tensor

TRANSITION = {
    "obs_state": Tensor("B *"),
    "action": Tensor("B *"),
    "reward": Tensor("B *", target=True),
    "terminated": Tensor("B *", target=True),
    "truncated": Tensor("B *", target=True),
    "episode_id": Tensor("B *", target=True),
}
ON_POLICY = {name: Tensor("B *") for name in ("log_prob", "value", "advantage", "return_")}
REPLAY = {name: Tensor("B *") for name in ("next_obs_state", "weight", "index", "n_step_discount")}
