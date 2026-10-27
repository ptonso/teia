from __future__ import annotations

from collections import namedtuple

import pytest
import torch
import yaml

from teia.core.module.net import TeiaNetModule
from teia.base.schema import ColumnMeta
from teia.node.data._frame import resolve_schema


class TokenDataModule:
    task = "cls"
    label_names: list = []
    num_labels = 5
    num_classes = 5
    num_targets = 2

    def batch_meta(self) -> dict[str, tuple[int, ...]]:
        return {"x": (4,), "target": (3, 2)}

    def batch_type(self):
        return namedtuple("Batch", ["x", "target"])


def test_schema_yaml_uses_top_level_targets_not_column_roles(tmp_path):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "age": [30.5, 40.0],
            "segment": ["a", "b"],
            "price": [10.0, 20.0],
        }
    )
    columns = [ColumnMeta(name, str(frame[name].dtype)) for name in frame.columns]

    schema = resolve_schema(
        train_frame=frame,
        columns=columns,
        targets=["price"],
        task="reg",
        schema_path=tmp_path / "schema.yaml",
    )
    payload = yaml.safe_load((tmp_path / "schema.yaml").read_text())

    assert payload["targets"] == ["price"]
    assert all("role" not in column for column in payload["columns"].values())
    assert schema.groups["numerical"] == ["age"]
    assert "price" not in schema.input_columns()


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (["num_classes"], (1, 5)),
        (["num_targets"], (1, 2)),
        (["target", 3], (1, 3, 2, 3)),
        ([1], (1, 1)),
        ([32], (1, 32)),
    ],
)
def test_linear_head_resolves_out_key_tokens(shape: list[object], expected: tuple[int, ...]) -> None:
    module = TeiaNetModule(
        head={
            "_target_": "teia.node.net.one_to_one.agnostic.linear.LinearHead",
            "in": "batch.x", "out": {"pred.logits": shape},
        }
    )
    dm = TokenDataModule()

    module.configure_from_datamodule(dm)

    batch = dm.batch_type()(x=torch.zeros(1, 4), target=torch.zeros(1, 3, 2))
    assert tuple(module(batch).logits.shape) == expected


def test_unknown_out_key_token_fails_fast() -> None:
    module = TeiaNetModule(
        head={
            "_target_": "teia.node.net.one_to_one.agnostic.linear.LinearHead",
            "in": "batch.x", "out": {"pred.logits": ["missing"]},
        }
    )

    with pytest.raises(ValueError, match="unresolvable out_key meta-token 'missing'"):
        module.configure_from_datamodule(TokenDataModule())
