"""The reinforcement-learning presets compose, probe the contract's fields and act through the capture slice."""

from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from teia.core.instantiate import instantiate
from teia.core.runtime.composer import compose_project_config

TASKS = {
    "rl-discrete": {"obs_state_dim": 4, "num_actions": 2},
    "rl-continuous-replay": {"obs_state_dim": 3, "action_dim": 1},
    "rl-continuous": {"obs_state_dim": 3, "action_dim": 1},
}


def _build(project: Path, task: str):
    cfg = compose_project_config(project, overrides=[f"task={task}", "datamodule.env.num_envs=2"])
    contract = instantiate(cfg["task"])
    dm = instantiate({"_target_": "teia.core.datamodule.executor.TeiaDataModule", "_recursive_": False, "data_root": cfg["data_root"], **cfg["datamodule"]})
    dm.bind_contract(contract)
    module = instantiate({"_target_": "teia.core.module.net.TeiaNetModule", "_recursive_": False, **cfg["netmodule"]})
    dm.bind_module(module)
    module.configure_from_datamodule(dm)
    module.dry_run(dm)
    return contract, dm, module


@pytest.mark.parametrize("task", TASKS)
def test_the_datamodule_probes_every_field_the_contract_declares(task: str, tmp_path: Path) -> None:
    contract, dm, _ = _build(tmp_path, task)
    assert set(contract.batch) <= set(dm.batch_type()._fields)
    contract.check_data(dm.batch_meta(), dm.meta(), dm.batch_type()._fields)


@pytest.mark.parametrize("task", TASKS)
def test_engine_dims_reach_the_contract_meta(task: str, tmp_path: Path) -> None:
    _, dm, _ = _build(tmp_path, task)
    assert {key: dm.meta()[key] for key in TASKS[task]} == TASKS[task]


@pytest.mark.parametrize("task", TASKS)
def test_acting_returns_the_atoms_the_contract_captures(task: str, tmp_path: Path) -> None:
    contract, dm, module = _build(tmp_path, task)
    assert set(module.act_forward(dm._sample_batch(), deterministic=False)) == set(contract.capture)


def test_sac_acting_never_runs_the_critics(tmp_path: Path) -> None:
    _, dm, module = _build(tmp_path, "rl-continuous-replay")
    called = []
    for name in ("q1", "q2", "q1_target", "q2_target"):
        getattr(module, name).register_forward_hook(lambda *args, name=name: called.append(name))
    module.act_forward(dm._sample_batch(), deterministic=True)
    assert called == []


def test_ppo_acting_runs_the_value_head_because_the_contract_captures_it(tmp_path: Path) -> None:
    _, dm, module = _build(tmp_path, "rl-continuous")
    called = []
    module.value_head.register_forward_hook(lambda *args: called.append("value"))
    module.act_forward(dm._sample_batch())
    assert called == ["value"]


def test_sac_target_entropy_follows_the_action_dimension(tmp_path: Path) -> None:
    _, _, module = _build(tmp_path, "rl-continuous-replay")
    assert module.alpha_loss.target_entropy == -1.0
