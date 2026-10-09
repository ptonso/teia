# Task Contract Spec

Related: **Read first** [batch](batch.md), [eval/nodes](eval/nodes.md). **See also** [core/task](../core/task.md), [core/module/capture_map](../core/module/capture_map.md), [core/eval](../core/eval.md), [core/config](../core/config.md).

## Overview

A task contract is a Python class that types the two seams between a run's three graphs:
- **Boundary A** is what the data graph hands the net graph: `batch.*` fields and `meta.*` dataset facts.
- **Boundary B** is what the net graph hands the eval graph: `capture.<route>.<atom>` atoms, plus the target `batch.*` fields captured as ground truth.

The contract says nothing about what happens inside a graph. Two networks, frozen teachers, inner optimization loops and multi-optimizer schedules are all invisible to it.

`teia.base.task` owns the base class and the spec types and ships no concrete contract. Concrete contracts live in the component library (`teia.task.*`). A run selects one through `task._target_`, and core validates the composed run against it without ever branching on which contract it is.

Walk-through:
1. At pre-flight, core instantiates `task._target_` and checks that the netmodule reads only declared `batch.*` and `meta.*` keys and fills every declared `capture` atom (`check_net`).
2. Also at pre-flight, core checks that every eval node reads only declared keys (`check_eval`).
3. After the data graph's plan stage, core checks the emitted field shapes and dtypes against the contract, with shape symbols bound from `meta` (`check_data`).
4. The capture writer reads the contract to decide how each atom is stored, and which `batch.*` fields to persist as ground truth.

Scenario: `VisionCls` declares:
- `batch.image` as `Tensor("B 3 H W", "float32")`;
- `batch.cls` as `Tensor("B", "int64", target=True)`;
- `meta.num_classes` and `meta.class_names`;
- `capture.cls.scores` as `Tensor("B num_classes", "float32")`.

Any datamodule emitting those fields, any netmodule mapping an activation onto `cls.scores`, and any evalmodule reading only those keys compose into a valid run.

## Language

- **task contract**: a `TaskContract` subclass. Its class path is the task's identity.
- **boundary A**: `batch.*` plus `meta.*`, the data→net seam.
- **boundary B**: `capture.*` plus target `batch.*`, `meta.*` and `log.*`, the net→eval seam.
- **spec**: the type of one boundary key (`Tensor`, `Ragged`, `Blob`, `Dim`, `Names`).
- **target field**: a `batch.*` field declared `target=True`. It is consumed by losses and captured as eval ground truth.
- **shape string**: space-separated dims. It starts with `B` (a `Tensor`'s batch dim) or `N` (a `Ragged` row dim). An integer is literal. A `meta` name expands to its value, and a tuple value such as `kpt_shape` fills several dims. `*` matches one dim of any extent, or any number of dims when it is last. Any other name is free but must agree wherever it repeats, across all fields.
- **meta**: dataset-level facts resolved at the data graph's plan stage (`num_classes`, `class_names`, `kpt_shape`), exposed by `TeiaDataModule.meta()` ([core/datamodule](../core/datamodule.md)).
- **log key**: `log.<column>`, a per-epoch logger series read by the eval graph ([core/eval](../core/eval.md)).

## Map

- `teia.base.task`: `TaskContract`, the spec types, and the three checks.
- Concrete contracts: `teia.task.*` in the component library, one module per task.
- Consumers: pre-flight in `teia.core.runtime.validate`, the post-plan check in `teia.core.runtime.engine`, and the capture writer in `teia.core.capture`.

## Contracts

### Spec types

```python
@dataclass(frozen=True)
class Tensor:                      # one row per sample; leading dim B
    shape: str                     # "B 3 H W", "B num_classes", "B"
    dtype: str | None = None       # torch dtype name; None = any
    target: bool = False

@dataclass(frozen=True)
class Ragged:                      # rows across samples, flat [N, ...]
    shape: str                     # "N 4"
    index: str                     # sibling key holding each row's sample index
    dtype: str | None = None
    target: bool = False

@dataclass(frozen=True)
class Blob:                        # one file per sample: mask, image, media
    format: str                    # "png" | "gif" | "mp4"
    target: bool = False

@dataclass(frozen=True)
class Dim: ...                     # meta: an int or an int tuple

@dataclass(frozen=True)
class Names:                       # meta: a list of labels
    dim: str                       # meta name giving its length
```

For a `Ragged` batch field, `index` names a batch field (`batch_idx`). For a `Ragged` capture atom, it names an atom of the same route (`det.sample_idx`). The writer offsets the index by the number of samples already written, so the stored index is global across the split.

### TaskContract

```python
class TaskContract:
    meta: ClassVar[dict[str, Dim | Names]] = {}
    batch: ClassVar[dict[str, Tensor | Ragged | Blob]] = {}   # keyed by field name, no "batch." prefix
    capture: ClassVar[dict[str, Tensor | Ragged | Blob]] = {} # keyed by "route.atom"

    def targets(self) -> list[str]: ...          # batch fields with target=True
    def check_net(self, in_keys: set[str], capture_map: Mapping[str, str]) -> None: ...
    def check_eval(self, in_keys: set[str]) -> None: ...
    def check_data(self, batch_meta: Mapping[str, tuple[int, ...]], meta: Mapping[str, Any], fields: Iterable[str]) -> None: ...
```

- **`check_net`**:
  - every `batch.*` and `meta.*` key the netmodule reads is declared;
  - every declared `capture` atom has an entry in the netmodule's `capture:` map ([core/module/capture_map](../core/module/capture_map.md)).

  A netmodule may capture extra atoms, but eval modules cannot rely on them.
- **`check_eval`**: every eval `in` key that is not an `eval.*` key is a declared `capture` atom, a target `batch.*` field, a declared `meta.*` key, or a `log.*` key.
- **`check_data`**:
  - every declared `meta` key is published, with each `Names` list as long as its `dim`;
  - every declared batch field is among `fields`, the composed batch's field names;
  - every declared `Tensor` or `Ragged` field that `batch_meta` reports has a matching shape after the leading `B` or `N`.

  It raises on the first mismatch. List fields (absent from `batch_meta`) and `Blob` fields are presence-checked only. Dtypes are enforced when atoms are captured, not here.

A check raises `TaskContractError`. The message names the contract class, the key, and what was expected and found.

### Contract authoring

```python
# teia/task/vision_cls.py
from teia.base.task import Dim, Names, TaskContract, Tensor

class VisionCls(TaskContract):
    """Single-label image classification."""
    meta = {"num_classes": Dim(), "class_names": Names("num_classes")}
    batch = {"image": Tensor("B 3 H W", "float32"), "cls": Tensor("B", "int64", target=True)}
    capture = {"cls.scores": Tensor("B num_classes", "float32")}
```

A variant subclasses its parent and replaces only what differs. For example, a distillation contract replaces `capture` with `student.scores` and `teacher.scores` and keeps the parent's boundary A, so the parent's datamodules fit it unchanged.

## Extending

To add a task:
1. Write one `TaskContract` subclass in `teia.task.<task>`.
2. Reuse field names from [batch](batch.md) before adding new ones.
3. Pick the most specific spec type for each key: `Ragged` for per-instance rows, and `Blob` for anything too large to keep as a numeric column.
4. Then write the task's modules and preset ([core/task](../core/task.md)).

## Constraints

- `teia.base.task` MUST NOT ship a concrete contract and MUST NOT import `teia.node` or `teia.core`.
- `teia.core` MUST NOT import a concrete contract or branch on a contract's class. It instantiates `task._target_` and calls the three checks only.
- A contract MUST describe only the two boundaries. It MUST NOT name `feat.*`, `pred.*`, `act.*`, `post.*`, `loss.*` or `eval.*` keys.
- Every `capture` key MUST have the form `route.atom`. Every `Ragged.index` MUST name a key declared in the same section, with the same `target` flag.
- Every shape symbol that is not `B`, `N`, `*` or a literal MUST be either a declared `meta` key or a free name that repeats consistently.
- Contract checks MUST be covered by `tests/test_task_contract.py`, and every shipped task by `tests/test_task_conformance.py`.

