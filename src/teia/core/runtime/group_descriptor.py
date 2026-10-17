"""``group.yaml``: the identity of a replicated train (a group of ``rep_<i>`` run directories).

See specs/core/runtime.md.
"""

from __future__ import annotations

from pathlib import Path

from teia.core.utils import read_yaml, write_yaml

__all__ = ["is_group_dir", "write_group_descriptor", "replica_dirs"]

_FILENAME = "group.yaml"


def is_group_dir(path: str | Path) -> bool:
    return (Path(path) / _FILENAME).is_file()


def write_group_descriptor(group_dir: str | Path, *, n_runs: int, base_seed: int, status: str) -> Path:
    payload = {
        "n_runs": n_runs,
        "base_seed": base_seed,
        "replicas": [f"rep_{i}" for i in range(n_runs)],
        "status": status,
    }
    return write_yaml(Path(group_dir) / _FILENAME, payload)


def replica_dirs(group_dir: str | Path) -> list[Path]:
    """The replica run directories of a group in replica order; each must exist."""
    group_dir = Path(group_dir)
    if not is_group_dir(group_dir):
        raise FileNotFoundError(f"{group_dir} is not a group directory (no {_FILENAME}).")
    dirs = [group_dir / name for name in read_yaml(group_dir / _FILENAME)["replicas"]]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"group {group_dir} lacks replica directories: {missing}.")
    return dirs
