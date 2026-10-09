"""Eval-node contracts, the eval-plane counterpart of ``teia.base.data.IoNode`` and
``teia.base.net.TeiaNode``.

An eval graph is built from three node kinds, ``Metric``, ``View`` and ``Comparison``, wired to
each other by ``in_key``/``out_key`` over the eval workspace namespaces ``capture.*``,
``batch.*`` and ``eval.*``. ``teia.base`` owns only these ABCs; concrete components live in
``teia.node.eval.*`` (the ``teia`` distribution).

See packages/teia/specs/base/eval/nodes.md for the full contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EvalNode(ABC):
    """Base contract shared by every eval node.

    Concrete components are constructed from flat kwargs. The graph runner sets the envelope
    (``in_key``/``out_key``) as attributes after instantiation, the same way the io and nn
    graphs bind their own envelopes.
    """

    in_key: list[str]
    out_key: list[str]

    #: Extra ``from_source`` kwargs this node takes beyond its ``in`` keys. A graph-aware metric
    #: (see ``Metric.after_pass``) reruns the nodes that declare the kwarg it supplies.
    ACCEPTS: frozenset[str] = frozenset()


class Metric(EvalNode):
    """One implementation, two drivers.

    ``from_source`` is the materialized driver: it reads a finished capture store and returns
    whole-split results. It is required. ``update``/``compute``/``reset`` are the optional
    streaming driver, called once per batch during fit-validation. A metric that does not
    override ``update`` cannot be used as a val metric; that is the only signal, there is no
    separate flag, and the eval-graph validator rejects such a wiring at pre-flight.
    """

    #: Monitor names this metric publishes, relative to the split (e.g. ``("macro_f1",)``).
    #: Empty when the metric produces only array results consumed by views.
    MONITORS: tuple[str, ...] = ()

    #: ``_target_`` path(s) of Lightning callback(s) this metric needs present in
    #: ``callbacks.items`` for meaningful output (e.g. a metric reading ``lr-*`` CSV columns
    #: needs ``lightning.pytorch.callbacks.LearningRateMonitor``). Checked at pre-flight
    #: (``core/runtime/validate.py::_validate_eval_callbacks``); empty when the metric needs
    #: nothing beyond its declared ``in`` keys.
    REQUIRES_CALLBACK: tuple[str, ...] = ()

    @abstractmethod
    def from_source(self, source: "EvalSource", **inputs: Any) -> dict[str, Any]:
        """Materialized driver: read the capture store, return whole-split results keyed by
        ``out_key``."""

    def after_pass(self, graph: Any) -> None:
        """Optional hook, called once per metric after the whole graph ran. ``graph`` is the runner's
        ``EvalPass`` handle (``records``, ``results``, ``out_dir``, ``downstream``, ``rerun``), which
        lets a metric rerun other nodes with extra kwargs into another folder. Default: nothing."""
        return None

    def update(self, **inputs: Any) -> None:
        """Streaming driver: accumulate one validation batch. Override to make the metric
        usable as a val metric."""
        raise NotImplementedError(f"{type(self).__name__} has no streaming driver (no update()).")

    def compute(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} has no streaming driver (no compute()).")

    def reset(self) -> None:
        return None


class View(EvalNode):
    """Terminal: consumes ``eval.*`` results, writes artifact files, declares no ``out_key``."""

    @abstractmethod
    def render(self, out_dir: Path, **inputs: Any) -> list[Path]:
        """Consume named results, write artifact files, return what was written."""


class Comparison(EvalNode):
    """Cross-run node: many finished runs in, one artifact out (comparison tables, curve
    overlays)."""

    @abstractmethod
    def compare(self, sources: list["EvalSource"], out_dir: Path) -> list[Path]:
        """Read every source run, write the comparison artifact, return what was written."""


class EvalSource(ABC):
    """A read handle over one finished run directory: capture store, config snapshot,
    per-epoch logger CSV, run descriptor. Never exposes a live ``LightningModule``,
    datamodule, trainer, or dataloader."""

    run_dir: Path

    @abstractmethod
    def routes(self) -> list[str]:
        """Capture routes present in the manifest."""

    @abstractmethod
    def meta(self, key: str, default: Any = None) -> Any:
        """A manifest ``meta`` value (``class_names``, ...)."""

    @abstractmethod
    def has(self, key: str) -> bool:
        """Whether ``capture.<route>.<atom>`` (or ``batch.<atom>``) is present."""

    @abstractmethod
    def column(self, key: str) -> Any:
        """The column's data: memmap / row lane / blob paths."""

    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        """``config/composed.yaml``."""

    @abstractmethod
    def curves(self) -> Any:
        """``logs/*/metrics.csv``, per-epoch."""

    @abstractmethod
    def descriptor(self) -> Mapping[str, Any]:
        """``artifacts/run.yaml``."""

    @abstractmethod
    def overrides(self) -> list[str]:
        """``config/overrides.yaml``'s ``overrides`` list: the raw CLI/API overrides this run
        was composed with.

        Cross-run comparison (``factor_diff``) uses this to derive which config axis varies
        between sibling runs. The fully-composed ``config()`` cannot answer that, since every
        run's composed config differs in thousands of resolved defaults, not just the one axis
        that was swept.
        """


__all__ = ["EvalNode", "Metric", "View", "Comparison", "EvalSource"]
