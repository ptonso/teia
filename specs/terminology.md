# Domain Terminology Spec

Related: **Read first** [architecture](architecture.md). **See also** [data/layout](data/layout.md), [net/layout](net/layout.md), [base/structure](base/structure.md), [base/task](base/task.md), [core/task](core/task.md), [core/eval](core/eval.md), [eval/layout](eval/layout.md).

## Overview

Domain words (`vision`, `tabular`, and siblings such as `audio`, or `set` used as a task label) name a **task choice**: which task, dataset or module a user picked. They are never an engine concept. Code is organized by two things instead:

- **structure**: the topology of the data (`set`, `sequence`, `grid`, `graph`), per [base/structure](base/structure.md).
- **contract**: what crosses a graph boundary, typed by the task contract's specs (`Tensor`, `Ragged`, `Blob`), per [base/task](base/task.md). The engine reads the spec type and never the meaning.

Task contracts are the one place a task name is code: `teia.task.vision_cls.VisionCls` is named for the task it types. They live only in `teia.task`, and no engine code imports them. A domain word found anywhere else in `teia.core`, `teia` or `teia.node` Python (a class name, a module name, a dict key used for dispatch) is naming debt to fix with this vocabulary, never a deliberate design choice.

## Language

- **task**: a named contract plus its module folders and preset ([core/task](core/task.md)). The name is a user choice.
- **route**: the first segment of a capture key (`capture.<route>.<atom>`), the namespace a task contract files predictions under.
- **structure**: the topology of a datum's domain.
- **recipe**: a config (task preset, module, dataset) where domain words are correct.

## Map

How each subsystem avoids domain words.

| Subsystem | Organized by | Where |
|---|---|---|
| data graph (`teia.node.data`) | node kind (`reader/transform/join/reshape/collate/writer`) plus `structure` (`Collate.structure`, `requires_structure`, `FIELD_VOCAB` typing fields `grid`, `sequence`, `set` or `None`). No domain folder. | [data/layout](data/layout.md) |
| net graph (`teia.node.net`) | `fan × assumes`. `one_to_one/{grid,set,…}/` holds function-named classes (`MLP`, `LinearHead`), and structure is conveyed by module location, never a domain word in a class name. | [net/layout](net/layout.md) |
| activation routing | atoms. An activation returns content-named atom dicts, and the netmodule's `capture:` map files them under the contract's routes. There is no prediction kind. | [capture_map](core/module/capture_map.md) |
| eval graph (`teia.node.eval`) | ABC kind × pairing. The layout spec forbids a domain folder and a domain-named class. | [eval/layout](eval/layout.md) |
| capture (`teia.core.capture`) | contract spec type. The lane and encoder follow `Tensor`/`Ragged`/`Blob` and the array dtype only. | [core/eval](core/eval.md) |
| tasks (`teia.task`, `conf/task`, `conf/<graph>module/<task>`) | the user's task choice. Contract classes, preset files and module folders carry task names correctly. | [core/task](core/task.md) |

## Contracts

### Triage for a found domain word

Work out which bucket an occurrence is in.

1. **Describing data structure** (a batch with a 2-D pixel grid, a flat unordered row of mixed-type fields): name it by structure, `grid`, `set`, `sequence` or `graph`. Structure lives in the module path or a `structure` or `requires_structure` attribute, never a domain-flavored class prefix.
2. **Describing what crosses a boundary** (boxes, a class distribution, a survival risk): name the atom by content (`boxes`, `scores`, `risk`) and type it in a task contract. Engine code reads only the spec type.
3. **The user's task or dataset choice**: leave it in config, or in a `teia.task` contract.

An occurrence that fits none of the three is naming debt.

### Route versus atom versus structure

- **Route** (`det`, `cls`, `student`) is chosen by the contract. It is the `capture.<route>.<atom>` prefix an evalmodule wires against.
- **Atom** (`boxes`, `score`, `category`) is one content-named quantity. Its contract spec decides how it is stored: a `Ragged` atom is flat rows plus an index, and a `Blob` atom is one file per sample.
- **Structure** (`grid`) organizes components, never capture or eval dispatch.

## Constraints

- A class, module or dispatch key in `teia.core`, `teia` or `teia.node` Python MUST NOT be named for a domain, except task contracts in `teia.task`.
- Dispatch MUST key on structure or a contract spec type, never on a task name, a contract class or a `domain` field.
- A graph word (`net`, `data`, `eval`) names both the code folder and the `node/<graph>` conf folder of that graph. The legacy words `nn` and `io` MUST NOT name a graph.
- Domain words MAY appear in config file and folder names, task names, dataset names and experiment names.
