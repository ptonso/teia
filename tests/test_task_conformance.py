"""Every shipped task composes with its contract, and `teia` ships no wiring beyond the machinery."""

from __future__ import annotations

from pathlib import Path

import importlib
import pkgutil

import pytest
import yaml

import teia.task
from teia.base.task import TaskContract
from teia.core.runtime.composer import compose_project_config
from teia.core.runtime.validate import _validate_task_contract

CONF = Path(__file__).parents[1] / "src" / "teia" / "conf"
GRAPHS = ("data", "net")
TASKS = sorted(path.stem for path in (CONF / "task").glob("*.yaml"))
MODULES = [(task, graph, path.stem) for task in TASKS for graph in GRAPHS for path in sorted((CONF / f"{graph}module" / task).glob("*.yaml"))]
EVALMODULES = sorted(f"{path.parent.name}/{path.stem}" for path in (CONF / "evalmodule").glob("*/*.yaml") if path.parent.name != "_shared")
ALLOWED_OPTIM = {"adam_1e3", "adamw_warmup_decay"}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.parametrize("task", TASKS)
def test_task_preset_passes_its_contract(task: str, project: Path) -> None:
    _validate_task_contract(compose_project_config(project, overrides=[f"task={task}"]))


@pytest.mark.parametrize(("task", "graph", "name"), MODULES)
def test_module_passes_the_contract_of_its_task(task: str, graph: str, name: str, project: Path) -> None:
    _validate_task_contract(compose_project_config(project, overrides=[f"task={task}", f"{graph}module={task}/{name}"]))


def _contracts() -> list[TaskContract]:
    found = []
    for info in pkgutil.iter_modules(teia.task.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"teia.task.{info.name}")
        found += [obj() for obj in vars(module).values() if isinstance(obj, type) and issubclass(obj, TaskContract) and obj.__module__ == module.__name__]
    return found


def _eval_in_keys(evalmodule: str, project: Path) -> set[str]:
    """The keys an evalmodule reads. Placeholders stand in for what it interpolates from other graphs (`data_root`, the forecast quantiles)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from teia.core.config.layers import resolve_conf_layout
    from teia.core.eval.graph import build_eval_graph, discover_node_entries
    from teia.core.eval.runner import split_evalmodule

    layout = resolve_conf_layout(project)
    searchpath = "[" + ",".join(layer.uri for layer in layout.searchpath) + "]"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(layout.primary.path), version_base="1.3"):
            config = compose(config_name="config", overrides=[f"hydra.searchpath={searchpath}", f"evalmodule={evalmodule}"])
    finally:
        GlobalHydra.instance().clear()
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, {"data_root": ".", "netmodule": {"act": {"quantiles": [0.5]}}})
    _, graph = split_evalmodule(OmegaConf.to_container(config.evalmodule, resolve=True))
    return {key for record in build_eval_graph(discover_node_entries(graph)) for key in record.in_key}


def _fits(evalmodule: str, contract: TaskContract, project: Path) -> bool:
    try:
        contract.check_eval(_eval_in_keys(evalmodule, project))
    except Exception:
        return False
    return True


@pytest.mark.parametrize("evalmodule", EVALMODULES)
def test_every_evalmodule_fits_a_shipped_contract(evalmodule: str, project: Path) -> None:
    assert any(_fits(evalmodule, contract, project) for contract in _contracts())


def test_every_shipped_contract_has_an_evalmodule(project: Path) -> None:
    orphans = [type(c).__name__ for c in _contracts() if not any(_fits(e, c, project) for e in EVALMODULES)]
    assert orphans == []


@pytest.mark.parametrize("task", TASKS)
def test_preset_selects_the_probe(task: str) -> None:
    defaults = yaml.safe_load((CONF / "task" / f"{task}.yaml").read_text())["defaults"]
    assert {"override /netmodule": f"{task}/linear_probe"} in defaults


def test_teia_ships_no_netmodule_but_the_probes() -> None:
    shipped = sorted(path.relative_to(CONF).as_posix() for path in (CONF / "netmodule").rglob("*.yaml"))
    assert all(path.endswith("/linear_probe.yaml") for path in shipped)
    assert "netmodule/_shared/linear_probe.yaml" in shipped


def test_teia_ships_no_tuned_system_recipe() -> None:
    assert not (CONF / "trainer").exists() and not (CONF / "callbacks").exists()
    assert {path.stem for path in (CONF / "optim").glob("*.yaml")} == ALLOWED_OPTIM
