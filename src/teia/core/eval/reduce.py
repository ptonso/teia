"""Cross-replica reduction of metric results: see specs/eval/reducer.md."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["reduce"]


def reduce(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold one ``results`` mapping per replica into one mapping with the keys of replica 0.

    A reduced dict keeps the shape of one replica's result and adds ``<field>_std`` (ddof=1) and
    ``<field>_values`` siblings, plus ``n_runs`` on each top-level result.
    """
    if not results:
        raise ValueError("reduce needs at least one replica result.")
    _require_same_keys([r.keys() for r in results], "results")
    reduced = {key: _fold([r[key] for r in results])[0] for key in results[0]}
    for value in reduced.values():
        if isinstance(value, dict):
            value["n_runs"] = len(results)
    return reduced


def _require_same_keys(key_sets: list[Any], where: str) -> None:
    first = set(key_sets[0])
    for keys in key_sets[1:]:
        if set(keys) != first:
            raise ValueError(f"replica key mismatch in {where}: {sorted(first ^ set(keys))}.")


def _fold(values: list[Any]) -> tuple[Any, dict[str, Any]]:
    if all(_is_number(v) for v in values):
        return _mean_std(np.asarray(values, dtype=float), values[0], scalar=True)
    arrays = [_numeric_array(v) for v in values]
    if all(a is not None for a in arrays) and len({a.shape for a in arrays}) == 1:
        return _mean_std(np.stack(arrays), values[0], scalar=False)
    if all(isinstance(v, dict) for v in values):
        return _fold_dicts(values), {}
    if _aligned_rows(values):
        return [_fold_dicts([v[i] for v in values]) for i in range(len(values[0]))], {}
    if all(isinstance(v, (str, bool, type(None))) for v in values) and all(v == values[0] for v in values):
        return values[0], {}
    return values[0], {"values": list(values)}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _numeric_array(value: Any) -> np.ndarray | None:
    if not isinstance(value, (np.ndarray, list, tuple)):
        return None
    try:
        array = np.asarray(value)
    except ValueError:
        return None
    return array if array.dtype.kind in "iuf" else None


def _mean_std(stacked: np.ndarray, first: Any, *, scalar: bool) -> tuple[Any, dict[str, Any]]:
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)
    if scalar:
        return float(mean), {"std": float(std), "values": stacked.tolist()}
    if isinstance(first, np.ndarray):
        return mean, {"std": std, "values": stacked}
    return mean.tolist(), {"std": std.tolist(), "values": stacked.tolist()}


def _fold_dicts(values: list[dict[str, Any]]) -> dict[str, Any]:
    _require_same_keys([v.keys() for v in values], "dict")
    out: dict[str, Any] = {}
    for key in values[0]:
        folded, siblings = _fold([v[key] for v in values])
        out[key] = folded
        for suffix, sibling in siblings.items():
            name = f"{key}_{suffix}"
            if name in values[0]:
                raise ValueError(f"reduced sibling '{name}' collides with an existing field.")
            out[name] = sibling
    return out


def _aligned_rows(values: list[Any]) -> bool:
    """List-of-dict tables of equal length whose string fields match row by row."""
    if not all(isinstance(v, list) and v and all(isinstance(row, dict) for row in v) for v in values):
        return False
    if len({len(v) for v in values}) != 1:
        return False
    for rows in zip(*values):
        if len({frozenset(row) for row in rows}) != 1:
            return False
        for key in rows[0]:
            column = [row[key] for row in rows]
            if all(isinstance(c, str) for c in column) and len(set(column)) != 1:
                return False
    return True
