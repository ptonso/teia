from __future__ import annotations

import numpy as np
import pytest

from teia.core.eval.reduce import reduce


def _rep(accuracy: float, **extra):
    return {"eval.cls": {"accuracy": accuracy, **extra}}


def test_scalar_mean_std_values_and_n_runs() -> None:
    out = reduce([_rep(0.91), _rep(0.93), _rep(0.92)])["eval.cls"]
    assert out["accuracy"] == pytest.approx(0.92)
    assert out["accuracy_std"] == pytest.approx(0.01)
    assert out["accuracy_values"] == [0.91, 0.93, 0.92]
    assert out["n_runs"] == 3


def test_one_replica_has_zero_std() -> None:
    out = reduce([_rep(0.5)])["eval.cls"]
    assert out["accuracy"] == 0.5
    assert out["accuracy_std"] == 0.0
    assert out["n_runs"] == 1


def test_array_and_nested_list_reduce_elementwise() -> None:
    out = reduce([_rep(0.0, cm=np.array([[1.0, 2.0]]), curve=[1, 2]), _rep(0.0, cm=np.array([[3.0, 4.0]]), curve=[3, 4])])["eval.cls"]
    np.testing.assert_allclose(out["cm"], [[2.0, 3.0]])
    np.testing.assert_allclose(out["cm_std"], np.std([[[1.0, 2.0]], [[3.0, 4.0]]], axis=0, ddof=1))
    assert out["curve"] == [2.0, 3.0]
    assert np.shape(out["curve_values"]) == (2, 2)


def test_aligned_rows_reduce_numeric_fields_per_row() -> None:
    a = [{"name": "cat", "f1": 0.5}, {"name": "dog", "f1": 0.7}]
    b = [{"name": "cat", "f1": 0.7}, {"name": "dog", "f1": 0.9}]
    rows = reduce([_rep(0.0, class_rows=a), _rep(0.0, class_rows=b)])["eval.cls"]["class_rows"]
    assert [r["name"] for r in rows] == ["cat", "dog"]
    assert rows[0]["f1"] == pytest.approx(0.6)
    assert rows[0]["f1_std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))


def test_unaligned_rows_pass_through() -> None:
    a = [{"name": "cat", "f1": 0.5}]
    b = [{"name": "dog", "f1": 0.7}]
    out = reduce([_rep(0.0, class_rows=a), _rep(0.0, class_rows=b)])["eval.cls"]
    assert out["class_rows"] == a
    assert out["class_rows_values"] == [a, b]


def test_equal_strings_kept_and_differing_pass_through() -> None:
    out = reduce([_rep(0.0, task="cls", note="a"), _rep(0.0, task="cls", note="b")])["eval.cls"]
    assert out["task"] == "cls"
    assert "task_values" not in out
    assert out["note"] == "a"
    assert out["note_values"] == ["a", "b"]


def test_ragged_curves_pass_through() -> None:
    out = reduce([_rep(0.0, x=[1, 2, 3]), _rep(0.0, x=[1, 2])])["eval.cls"]
    assert out["x"] == [1, 2, 3]
    assert out["x_values"] == [[1, 2, 3], [1, 2]]


def test_key_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        reduce([_rep(0.0, extra=1), _rep(0.0)])
    with pytest.raises(ValueError, match="key mismatch"):
        reduce([{"a": {"x": 1}}, {"b": {"x": 1}}])


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        reduce([])
