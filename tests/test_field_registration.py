"""Field registration is independent of the import order of a contract and of the collate that emits its fields."""

import subprocess
import sys

import pytest

from teia.base import fields


@pytest.fixture(autouse=True)
def _restore_vocab():
    vocab, provisional = dict(fields.FIELD_VOCAB), set(fields._PROVISIONAL)
    yield
    fields.FIELD_VOCAB.clear(), fields.FIELD_VOCAB.update(vocab)
    fields._PROVISIONAL.clear(), fields._PROVISIONAL.update(provisional)


def test_an_explicit_registration_replaces_a_contract_placeholder() -> None:
    fields.register_field("probe_field", provisional=True)
    fields.register_field("probe_field", structure="grid", optional=False)
    assert fields.FIELD_VOCAB["probe_field"].optional is False and fields.FIELD_VOCAB["probe_field"].structure == "grid"


def test_two_explicit_registrations_with_different_specs_conflict() -> None:
    fields.register_field("probe_field", structure="grid", optional=False)
    fields.register_field("probe_field", structure="grid", optional=False)
    with pytest.raises(ValueError, match="conflicts"):
        fields.register_field("probe_field", structure="set", optional=False)


def test_a_contract_declaring_a_collate_field_loads_in_either_order() -> None:
    for first, second in (("teia.task.vision_graph_node_cls", "teia.node.data.collate.graph"), ("teia.node.data.collate.graph", "teia.task.vision_graph_node_cls")):
        done = subprocess.run([sys.executable, "-c", f"import {first}, {second}"], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-500:]
