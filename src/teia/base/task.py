"""Task contracts: typed specs for the two graph boundaries (see teia:base/task.md).

Boundary A is ``batch.*`` + ``meta.*`` (data → net); boundary B is ``capture.<route>.<atom>`` plus the
target ``batch.*`` fields (net → eval). Concrete contracts live in ``teia.task``; core only
instantiates ``task._target_`` and calls the three checks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Union


class TaskContractError(ValueError):
    """A composed run violates its task contract."""


@dataclass(frozen=True)
class Tensor:
    """One row per sample; ``shape`` starts with ``B``."""

    shape: str
    dtype: str | None = None
    target: bool = False


@dataclass(frozen=True)
class Ragged:
    """Rows across samples, flat ``[N, ...]``; ``index`` names the sibling key holding each row's sample index."""

    shape: str
    index: str
    dtype: str | None = None
    target: bool = False


@dataclass(frozen=True)
class Blob:
    """One file per sample (mask, image, media) written in ``format``."""

    format: str
    target: bool = False


@dataclass(frozen=True)
class Dim:
    """Meta: an int or an int tuple."""


@dataclass(frozen=True)
class Names:
    """Meta: a list of labels whose length is the meta value ``dim``."""

    dim: str


Spec = Union[Tensor, Ragged, Blob]
MetaSpec = Union[Dim, Names]


class TaskContract:
    """Base task contract. Subclasses set ``meta``, ``batch`` (field → spec) and ``capture`` ("route.atom" → spec)."""

    meta: ClassVar[dict[str, MetaSpec]] = {}
    batch: ClassVar[dict[str, Spec]] = {}
    capture: ClassVar[dict[str, Spec]] = {}

    def targets(self) -> list[str]:
        """Batch fields captured as eval ground truth."""
        return [name for name, spec in self.batch.items() if getattr(spec, "target", False)]

    def derived(self) -> set[str]:
        """Capture atoms filled by ``CaptureMap`` itself: the ``index`` of a ragged capture atom."""
        return {spec.index for spec in self.capture.values() if isinstance(spec, Ragged)}

    def check_net(self, in_keys: Iterable[str], capture_map: Mapping[str, str]) -> None:
        """The netmodule reads only declared ``batch.*``/``meta.*`` keys and maps every non-derived capture atom."""
        for key in sorted(set(in_keys)):
            namespace, _, name = key.partition(".")
            if namespace == "batch" and name not in self.batch:
                self._fail(f"netmodule reads undeclared batch field '{key}'", self.batch)
            if namespace == "meta" and name not in self.meta:
                self._fail(f"netmodule reads undeclared meta key '{key}'", self.meta)
        missing = sorted(set(self.capture) - self.derived() - set(capture_map))
        if missing:
            self._fail(f"netmodule `capture:` map lacks {missing}", self.capture)

    def check_eval(self, in_keys: Iterable[str]) -> None:
        """Every eval ``in`` key is a declared capture atom, target field, meta key, ``log.*`` or ``eval.*`` key."""
        targets = set(self.targets())
        for key in sorted(set(in_keys)):
            namespace, _, name = key.partition(".")
            known = {
                "capture": name in self.capture,
                "batch": name in targets,
                "meta": name in self.meta,
                "log": True,
                "eval": True,
            }.get(namespace, False)
            if not known:
                self._fail(f"evalmodule reads '{key}', which the contract does not declare", None)

    def check_data(self, batch_meta: Mapping[str, tuple[int, ...]], meta: Mapping[str, Any], fields: Iterable[str]) -> None:
        """Every declared meta key is published, every declared batch field is emitted (``fields``,
        the composed batch's field names), and every tensor field in ``batch_meta`` matches its shape.

        ``batch_meta`` excludes the batch (or ragged row) dim and omits non-tensor (list) fields, so
        those and ``Blob`` fields are presence-checked only. An empty ``batch_meta`` (a datamodule that
        opts out of shapes) skips the shape checks.
        """
        for name, spec in self.meta.items():
            if name not in meta:
                self._fail(f"datamodule does not publish meta '{name}'", self.meta)
            if isinstance(spec, Names) and len(meta[name]) != _as_int(meta[spec.dim]):
                self._fail(f"meta '{name}' has {len(meta[name])} names but '{spec.dim}' is {meta[spec.dim]}", None)
        emitted = set(fields)
        for name in self.batch:
            if name not in emitted:
                self._fail(f"datamodule does not emit batch field '{name}'", self.batch)
        bound: dict[str, int] = {}
        for name, spec in self.batch.items():
            if isinstance(spec, Blob) or name not in batch_meta:
                continue
            lead = "B" if isinstance(spec, Tensor) else "N"
            tokens = spec.shape.split()
            if not tokens or tokens[0] != lead:
                self._fail(f"batch field '{name}' shape '{spec.shape}' must start with '{lead}'", None)
            self._match(name, tokens[1:], tuple(batch_meta[name]), meta, bound)

    def _match(
        self, name: str, tokens: list[str], dims: tuple[int, ...], meta: Mapping[str, Any], bound: dict[str, int]
    ) -> None:
        expected: list[int | None] = []
        for i, token in enumerate(tokens):
            if token == "*" and i == len(tokens) - 1:
                expected.extend([None] * max(len(dims) - len(expected), 0))
                break
            if token == "*":
                expected.append(None)
            elif token.isdigit():
                expected.append(int(token))
            elif token in self.meta:
                value = meta[token]
                expected.extend(int(v) for v in (value if isinstance(value, (tuple, list)) else [value]))
            else:
                expected.append(bound.setdefault(token, dims[len(expected)] if len(expected) < len(dims) else -1))
        if len(expected) != len(dims) or any(e is not None and e != d for e, d in zip(expected, dims)):
            self._fail(f"batch field '{name}' has shape {dims}, contract expects '{' '.join(tokens)}' → {expected}", None)

    def _fail(self, message: str, declared: Mapping[str, Any] | None) -> None:
        hint = f" (declared: {sorted(declared)})" if declared else ""
        raise TaskContractError(f"{type(self).__module__}.{type(self).__name__}: {message}{hint}.")


def _as_int(value: Any) -> int:
    return int(value[0]) if isinstance(value, (tuple, list)) else int(value)


__all__ = ["Blob", "Dim", "Names", "Ragged", "TaskContract", "TaskContractError", "Tensor"]
