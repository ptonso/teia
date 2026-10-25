import subprocess
import sys

import pytest

from teia.base.fields import FIELD_VOCAB
from teia.base.task import Dim, TaskContract, TaskContractError, Tensor


class Continuous(TaskContract):
    meta = {"obs_dim": Dim()}
    batch = {"fb_obs": Tensor("B *"), "fb_action": Tensor("B *"), "fb_reward": Tensor("B *", target=True)}
    capture = {"policy.action": Tensor("B *"), "policy.value": Tensor("B")}
    feedback = {"policy.action": "fb_action", "policy.value": "fb_reward"}


def test_a_contract_registers_its_batch_fields_when_declared():
    assert {"fb_obs", "fb_action", "fb_reward"} <= set(FIELD_VOCAB)
    assert FIELD_VOCAB["fb_obs"].optional


def test_core_vocabulary_names_no_rl_field():
    code = "from teia.base.fields import FIELD_VOCAB as V; print(sorted({'reward','terminated','truncated','obs_state','next_obs_state','episode_start','mask'} & set(V)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_feedback_must_return_the_policy_action():
    class NoAction(Continuous):
        feedback = {"policy.value": "fb_reward"}

    with pytest.raises(TaskContractError, match="policy.action"):
        NoAction().check_net(["batch.fb_obs"], {"policy.action": "act.action", "policy.value": "pred.value"})


def test_feedback_atoms_and_fields_must_be_declared():
    class BadAtom(Continuous):
        feedback = {"policy.action": "fb_action", "policy.entropy": "fb_reward"}

    class BadField(Continuous):
        feedback = {"policy.action": "missing"}

    capture_map = {"policy.action": "act.action", "policy.value": "pred.value"}
    with pytest.raises(TaskContractError, match="capture atom"):
        BadAtom().check_net(["batch.fb_obs"], capture_map)
    with pytest.raises(TaskContractError, match="batch field"):
        BadField().check_net(["batch.fb_obs"], capture_map)


def test_a_valid_feedback_passes():
    Continuous().check_net(["batch.fb_obs", "meta.obs_dim"], {"policy.action": "act.action", "policy.value": "pred.value"})
