# Structure Spec

Related: **Read first** [batch](batch.md). **See also** [data/nodes](data/nodes.md), [net/nodes](net/nodes.md), [core/data_graph](../core/data_graph.md).

## Overview

Structure is the topology of a signal's domain, expressed as a small type algebra of `set`, `sequence`, `grid` and `graph`. It is the single axis on which both data and compute are organized: it types every `batch.*` field at the data↔net seam, and it is half of the `assumes` axis that places net components (`node/<fan>/<assumes>/`).

A modality name (vision, audio, text) is a bundle of structure, value-type and format. The system organizes by structure and treats value-type and format as orthogonal attributes. One grid encoder therefore serves an image, a mel-spectrogram and a video frame.

Walk-through: an data graph reads bytes in some format and emits `batch.*` fields, each carrying one structural type. The net graph's dry run checks that each consuming node accepts the structure of the field it reads. The value-type of the field picks the first trunk node (the embedder); the structure picks the encoder family that follows.

Scenario: a time-series regression reads a `sequence` of continuous values and predicts a scalar. It is one structured input plus a regression objective, never "mixed modality".

## Language

- **signal**: a datum viewed as a function over an index domain.
- **structure**: the topology of that domain, built from primitive axis-symmetries.
- **primitive**: one of `set` (unordered), `sequence` (ordered), `graph` (relational).
- **composite structure**: a product or nesting of primitives (`grid` is a product of ordered axes, `video` is a sequence of grids).
- **value-type**: what each domain point carries, `continuous` or `categorical`.
- **modeling freedom**: the choice of which symmetries a trunk exploits over a fixed structural type.
- **assumes**: value-type × structure, the second level of the nn Python tree.

## Map

- `teia.base` owns the structure vocabulary and value-type. `FIELD_VOCAB` (`teia.base.fields`) types each field.
- Concrete encoders and embedders live in `teia.node.net.*`, placed by `assumes`.
- Field vocabulary: [batch](batch.md). Nodes that consume it: [net/nodes](net/nodes.md).

## Contracts

### Structural type

The metadata a `batch.*` field carries, consumed by the dry run, the embedder and the encoder:

- `primitive` is one of `set`, `sequence`, `grid`, `graph`.
- `axes` is an ordered list of `(symmetry, extent?)`. A `grid` is `[ordered, ordered, …]`, a `sequence` is `[ordered]`, a `set` is `[unordered]`. Composites nest.
- `value_type` is `continuous` (with `channels`) or `categorical` (with `cardinality`).

The symmetry hierarchy is `graph ⊃ grid ⊃ sequence` and `graph ⊃ set`.

### One canonical type per datum

Apparent ambiguity (video as a 3-D grid or a sequence of grids, a relational database as a graph or a flat set) is modeling freedom over one type, not a second type.

### Canonical mappings

Derivations, not enum members:

| Signal | structure | value-type |
|---|---|---|
| image | grid (2-D) | continuous, C channels |
| mel-spectrogram | grid (2-D) | continuous, 1 channel |
| audio waveform | sequence | continuous |
| text / DNA | sequence | categorical |
| tabular row | set | mixed (per field) |
| video | sequence of grids | continuous |
| relational DB | graph of sets | mixed |

### Compute mapping

An encoder declares the structure it consumes, and `teia.node.net.<fan>.<structure>` serves that structure whatever format produced it. The **embedder** maps `(raw element, value-type)` to element vectors before the structure encoder (see [net/nodes](net/nodes.md)). Composite encoders are compositions.

### Set field slots

A `set` with heterogeneous fields refines value-type into per-field slots. The five core slots are `numerical`, `categorical`, `count`, `temporal` and `boolean`. Extension slots (`text`, `ordinal`, `high-cardinality`) are declared only when needed. Each slot resolves to exactly one coarse `value_type`, and the slot picks the embedder variant.

## Constraints

- A `batch.*` field MUST carry a structural type, and the dry run rejects a structural mismatch between a field and its consumer.
- Structure MUST NOT encode format or modality. A `grid` is a `grid` whether it came from a JPEG or an STFT.
- `teia.base` MUST own only the structure vocabulary and value-type. Concrete encoders and embedders live in `teia.node.net.*`.
- Structure stays orthogonal to the objective.

