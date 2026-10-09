"""``CaptureMap``: the single producer of ``capture.*`` atoms (teia:core/module/capture_map.md).

A step's activated tensors (``act.*``) and per-sample postprocess dicts (``post.<alias>.<atom>``)
are read through the netmodule's ``capture:`` map and assembled by the task contract's spec type:
``Tensor`` stacks, ``Ragged`` concatenates and derives its index atom, ``Blob`` stays per sample.
The contract's target ``batch.*`` fields are added from the same batch. Without a contract,
``act.*`` sources are tensors, ``post.*`` sources are ragged with a derived ``<route>.sample_idx``,
and targets are the ``batch.*`` keys the evalmodule reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from teia.base.task import Blob, Ragged, TaskContract, Tensor

__all__ = ["CaptureMap"]


class CaptureMap:
    def __init__(
        self,
        capture_map: Mapping[str, str],
        activation_outputs: Mapping[str, str],
        contract: TaskContract | None,
        eval_batch_keys: Iterable[str] = (),
    ) -> None:
        """``activation_outputs`` maps each ``act.*`` out-key to the activation alias producing it."""
        self.sources = {str(k): str(v) for k, v in capture_map.items()}
        self.activation_outputs = dict(activation_outputs)
        self.contract = contract
        if contract is None:
            self.specs = {
                atom: Tensor("B *") if source.startswith("act.") else Ragged("N *", index=f"{atom.split('.')[0]}.sample_idx")
                for atom, source in self.sources.items()
            }
            self.targets = sorted(k.removeprefix("batch.") for k in eval_batch_keys if k.startswith("batch."))
            self.target_specs: dict[str, Any] = {}
        else:
            self.specs = dict(contract.capture)
            self.targets = contract.targets()
            self.target_specs = {name: contract.batch[name] for name in self.targets}
        for atom, source in self.sources.items():
            if atom not in self.specs:
                raise ValueError(f"capture atom '{atom}' is not declared by the task contract {type(contract).__name__}.")
            if source.startswith("act."):
                if source not in self.activation_outputs:
                    raise ValueError(f"capture '{atom}: {source}': no activation declares out-key '{source}'.")
            elif source.startswith("post."):
                alias = source.split(".")[1]
                if alias not in set(self.activation_outputs.values()):
                    raise ValueError(f"capture '{atom}: {source}': '{alias}' is not an activation node.")
            else:
                raise ValueError(f"capture '{atom}: {source}': a source must be an `act.*` or `post.<alias>.<atom>` key.")

    def post_aliases(self) -> set[str]:
        """Activation aliases whose ``postprocess`` the map needs."""
        return {source.split(".")[1] for source in self.sources.values() if source.startswith("post.")}

    def atoms(
        self,
        batch: Any,
        activated: Mapping[str, Mapping[str, Any]],
        post: Mapping[str, list[dict]],
        batch_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """One step's atoms: ``{"capture.<route>.<atom>": value, "batch.<field>": value}``.

        ``batch_keys`` overrides which ``batch.*`` fields are added (default: the contract targets);
        inference passes the fields its writer nodes read.

        A ``Tensor``/``Ragged`` value is a CPU tensor with leading row dim; a ``Blob`` value is a
        list with one array per sample. A ragged atom's index holds batch-local sample positions;
        the store offsets it to be global across the split.
        """
        out: dict[str, Any] = {}
        for atom, source in self.sources.items():
            spec = self.specs[atom]
            if source.startswith("act."):
                value = activated[self.activation_outputs[source]][source.removeprefix("act.")]
                out[f"capture.{atom}"] = list(_cpu(value)) if isinstance(spec, Blob) else _cpu(value)
                continue
            _, alias, name = source.split(".", 2)
            rows = [sample[name] for sample in post[alias]]
            if isinstance(spec, Ragged):
                out[f"capture.{atom}"] = _concat(rows)
                out[f"capture.{spec.index}"] = torch.cat(
                    [torch.full((_count(row),), i, dtype=torch.int64) for i, row in enumerate(rows)]
                ) if rows else torch.zeros(0, dtype=torch.int64)
            elif isinstance(spec, Blob):
                out[f"capture.{atom}"] = [_cpu(row) for row in rows]
            else:
                out[f"capture.{atom}"] = torch.stack([torch.as_tensor(np.asarray(row)) for row in rows]) if rows else torch.zeros(0)
        fields = self.targets if batch_keys is None else [k.removeprefix("batch.") for k in batch_keys]
        for field in fields:
            value, spec = getattr(batch, field), self.target_specs.get(field)
            if isinstance(spec, Blob):
                value = list(_cpu(value))
            elif isinstance(spec, Ragged) and isinstance(value, list):
                value = [np.asarray(_cpu(item)) for sample in value for item in sample]  # per-sample lists → instance rows
            out[f"batch.{field}"] = _cpu(value)
        return out


def _cpu(value: Any) -> Any:
    """Detach to CPU; half/bfloat16 widen to float32 (numpy has no bfloat16)."""
    if not isinstance(value, torch.Tensor):
        return value
    value = value.detach().cpu()
    return value.float() if value.dtype in (torch.bfloat16, torch.float16) else value


def _count(row: Any) -> int:
    return len(row) if isinstance(row, list) else int(np.asarray(row).shape[0]) if np.ndim(row) else 1


def _concat(rows: list[Any]) -> Any:
    """Concatenate per-sample row blocks; lists of variable-size items (polygons) stay a flat list."""
    if any(isinstance(row, list) for row in rows):
        return [np.asarray(item) for row in rows for item in row]
    arrays = [np.atleast_1d(np.asarray(row)) for row in rows]
    return torch.as_tensor(np.concatenate(arrays)) if arrays else torch.zeros(0)
