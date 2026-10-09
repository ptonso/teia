"""IO-node contracts — the data-plane analog of ``teia.base.net.TeiaNode``.

Six atomic node kinds (``Reader``/``Transform``/``Join``/``Reshape``/``Collate``/``Writer``) wired by
``in_key``/``out_key`` over the io workspace (``stream.*`` → ``item.*`` → ``batch.*``). ``teia.base`` owns
only the ABCs, the phase/stage vocabulary, and the key typing; concrete components live in
``teia.node.data.*``. See teia:base/data/nodes.md.
"""

from __future__ import annotations

from pathlib import Path

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Callable, ClassVar, Literal, get_args

Phase = Literal["source", "preprocess", "augment", "target", "collate", "writer"]
Stage = Literal["plan", "item", "stream", "batch"]
Kind = Literal["read", "transform", "join", "reshape", "collate", "write"]

PHASES: frozenset[str] = frozenset(get_args(Phase))
STAGES: frozenset[str] = frozenset(get_args(Stage))
KINDS: frozenset[str] = frozenset(get_args(Kind))

STAGE_RANK: dict[Stage, int] = {"plan": 0, "item": 1, "stream": 1, "batch": 2}

_DEFAULT_PHASE: dict[Kind, Phase] = {
    "read": "source",
    "transform": "preprocess",
    "join": "target",
    "reshape": "preprocess",
    "collate": "collate",
    "write": "writer",
}
_DEFAULT_STAGE: dict[Kind, Stage] = {
    "read": "plan",
    "transform": "item",
    "join": "plan",
    "reshape": "plan",
    "collate": "batch",
    "write": "item",
}


def default_phase(kind: Kind) -> Phase:
    return _DEFAULT_PHASE[kind]


def default_stage(kind: Kind) -> Stage:
    return _DEFAULT_STAGE[kind]


Kernel = Callable[[Any, Mapping[str, Any]], tuple[Any, dict[str, Any]]]


class IoNode(ABC):
    """Base contract shared by every data node.

    Concrete components are constructed from flat kwargs; the executor injects the graph envelope
    (``in_key``/``out_key``/``phase``/``stage``) as attributes after instantiation — the way the module
    pipeline injects ``in_shape``/``out_shape`` into a ``TeiaNode``.
    """

    in_key: list[str]
    out_key: list[str]
    phase: Phase
    stage: Stage

    #: Build-time data contracts the executor validates before any IO (see core/datamodule graph
    #: ``_validate_contracts``). A node declares what its inputs must satisfy so a mis-wired graph
    #: fails fast instead of crashing mid-stream.
    requires_schema: bool = False
    """The graph must contain a producer of ``plan.schema`` (a schema-bound node needs ``ResolveSchema``)."""
    requires_structure: str | None = None
    """The upstream producer of this node's input must declare this signal structure (e.g. ``"grid"``);
    e.g. a geometric collate placed in a schema-only tabular graph is rejected at build."""

    kernel: ClassVar[Kernel | None] = None
    """The node's one implementation: a pure ``@staticmethod`` ``(value, params) -> (value, restore)``,
    torch, defined on the class. Export collects ``type(node).kernel``. ``None`` ⇒ boundary/exempt node.
    Any node on the sliced preprocess path from a model-input ``batch.*`` field back to its reader (a
    chain of ``preprocess`` ``Transform``s crossing one ``Collate`` seam) MUST set this or export fails
    fast naming the node."""

    def params(self) -> dict[str, Any]:
        """Plain, repr-round-trippable literals baked beside ``kernel`` at export (resolved sizes,
        fitted means/stds/vocabs)."""
        return {}

    def runtime_dims(self) -> dict[str, Any]:
        """Runtime dims this node derives from the data (e.g. ``{"num_classes": 80}``).

        The executor collects these after ``setup`` and exposes them as plain datamodule attributes,
        so a module head's ``out_key: [num_classes]`` resolves via ``getattr`` (see core/datamodule.md §4).
        """
        return {}

    def bind_dims(self, **dims: Any) -> None:
        """Receive the graph's published runtime dims post-setup (e.g. a multi-hot collate sizing
        its output to ``num_classes``). Default no-op."""
        return None

    def export_artifacts(self) -> dict[str, Any]:
        """JSON-able extras this node contributes to an export bundle, keyed by filename stem
        (e.g. a fitted schema/vocab). Collected from every node in the graph at export time
        (core/export.md). Default: none."""
        return {}


class Reader(IoNode):
    """``open(source) → iterate → key(item)`` emitting a uniform indexed stream. Emits ``stream.*``."""

    @abstractmethod
    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        """Yield ``(key, raw)`` for every sample in ``split``."""


class Transform(IoNode):
    """Deterministic (``preprocess``/``target``) or stochastic (``augment``) per-value map.

    A stateful transform sets ``stateful = True`` and implements ``fit(train_stream)`` (``plan`` stage)
    plus ``state``/``load_state`` for run-artifact (de)serialization; state fits on the train split only,
    unless ``fit_all_splits`` is set (dataset-metadata vocabularies — class lists, label maps — that must
    cover every split; this is not a leakage path since no fitted *statistic* crosses the split boundary).
    """

    stateful: bool = False
    fit_all_splits: bool = False

    def __call__(self, *inputs: Any) -> Any:
        fn = type(self).kernel
        if fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} defines no kernel; override __call__ or set a kernel."
            )
        return fn(inputs[0], self.params())[0]

    def fit(self, train_stream: Iterable[Any]) -> None:
        return None

    def state(self) -> Any:
        return None

    def load_state(self, state: Any) -> None:
        return None


class Join(IoNode):
    """Aligns streams by key (``how ∈ {inner, outer, left}``, ``on: key``); ``outer`` emits presence masks.

    ``__call__`` returns one row per aligned key: one raw per input (``None`` when absent), followed under
    ``how == "outer"`` by one ``bool`` present flag per input. The graph's ``out_key`` lists the raw keys
    then (outer only) the present keys, in input order.

    ``prefix`` re-exposes each collated field as ``<input>_<field>``: on for a multi-input fusion
    (camera + table), off for a pairing join whose inputs are parts of one sample (image ⨝ label file)."""

    how: str
    prefix: bool = False

    @abstractmethod
    def __call__(self, *streams: Any) -> Any: ...


class Reshape(IoNode):
    """Shared base: a node that redefines item granularity. See PlanReshape / StreamReshape."""

    def payload_spec(self) -> dict[str, Any]:
        """What an export client must send for a chain rooted here (e.g. window lookback/horizon/
        group/order). Default: none."""
        return {}


class PlanReshape(Reshape):
    """``plan``-stage stream restructuring over a materialized list (``window``/``group``/``explode``):
    called once, single-shot, stateless."""

    @abstractmethod
    def __call__(self, *inputs: Any) -> Any: ...


class StreamReshape(Reshape):
    """``stream``-stage stream restructuring over a live, unbounded source (rollout/replay buffers):
    incremental, stateful across calls, variable-cardinality output.

    Concrete buffers subclass this ABC; the contract is incremental and stateful, never
    a bulk load.
    """

    def configure(self, **kwargs: Any) -> None:
        """Bind run-time sizing + the live collater/generator once per fit. Each concrete buffer reads
        the subset of kwargs it needs (rollout: rollout_steps/num_envs/minibatch_size/update_epochs;
        replay: capacity/warmup_steps/train_frequency/...) and ignores the rest via ``**_``."""

    @abstractmethod
    def add(self, item: Any) -> None:
        """Append one item-workspace-shaped record."""

    @abstractmethod
    def ready(self) -> bool:
        """Whether a ready-cycle should emit now."""

    @abstractmethod
    def sample_iter(self) -> Iterator[Any]:
        """Yield the item-shaped records for one ready-cycle (each fed through the ordinary ``_collate_fn``)."""

    @abstractmethod
    def on_consumed(self) -> None:
        """Post-cycle hook — rollout: clear; replay: no-op."""

    def metrics(self) -> dict[str, float]:
        """Diagnostics (size, sample age, priority/reward statistics)."""
        return {}


class Collate(IoNode):
    """Assembles one atomic ``batch.<field>`` over the sample batch — the data↔module seam.

    ``field`` names a ``teia.base.fields.FIELD_VOCAB`` entry; ``structure`` is its declared signal geometry.
    """

    field: str
    structure: str | None = None

    def __call__(self, field_batch: list[Any]) -> Any:
        fn = type(self).kernel
        if fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} defines no kernel; override __call__ or set a kernel."
            )
        return fn(field_batch, self.params())[0]


class Writer(IoNode):
    """The inverse of a reader — materializes inference atoms back to a container (teia:core/infer.md).

    Called once per inference run with the whole split's values for each ``in`` key, in ``in``
    order (``capture.*`` atoms, ``batch.*`` fields, ``meta.*`` facts), and the destination dir."""

    @abstractmethod
    def __call__(self, *inputs: Any, dst: Path) -> list[Path]: ...
