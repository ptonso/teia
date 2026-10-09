# Inference Spec

Related: **Read first** [runtime](runtime.md), [data_graph](data_graph.md). **See also** [module/capture_map](module/capture_map.md), [export](export.md), [module/activation_route](module/activation_route.md).

## Overview

`teia infer` is the standalone, modality-agnostic prediction stage. Inference is a rewired data graph wired to the trained module: reader nodes over the new `--src`, the trained `preprocess` subgraph, the trained module, and writer nodes that materialize predictions to disk, plus the canonical `predictions.jsonl`. A single instance is a size-1 dataset.

It is dataset prediction only. Live collection is a separate stage and does not route through the infer runner.

Walk-through: teia loads the saved config, replays overrides, rebuilds the datamodule, and rebinds its readers to `--src`, dropping `target` and `augment` nodes by phase. It rebuilds the module and runs one Lightning `trainer.predict`, where `predict_step` runs `forward → activate → postprocess` over every activation node and reads the result through the netmodule's `capture:` map. The resulting `capture.*` atoms go to the datamodule's writer nodes and to `predictions.jsonl`.

## Language

- **canonical artifact**: the machine-readable `predictions.jsonl`.
- **prediction route**: the activation-node `forward → activate → postprocess` path shared by infer, capture and ONNX export, read through the `capture:` map ([module/capture_map](module/capture_map.md)).
- **infer graph**: the data graph rewired for inference, `source` plus `preprocess` over `--src` → trained module → `writer` nodes.
- **writer node**: the mirror of a reader ([base/data/nodes](../base/data/nodes.md)). It is a datamodule node whose `in` names the `capture.*` and `meta.*` keys it materializes (`in: [capture.cls.scores, meta.class_names]`), writing tabular `*_pred` columns, a `class_dir` tree or annotation files.
- **native mirror**: the annotated dataset a writer emits to `dst` in a chosen on-disk format.

## Map

- Runner: `teia.core.infer`, called from `infer_project` in [runtime](runtime.md).
- The datamodule protocol (`bind_infer_source`, `predict_dataloader`, `write_predictions`, `predict_ctx_builder`): [datamodule](datamodule.md).
- Atom extraction: [module/capture_map](module/capture_map.md).

## Contracts

### Inputs and outputs

Inputs: `--run-dir`, `--checkpoint` override, `--src` (folder or single file), `--dst` (alias of `--output-dir`), and the composed `infer` config. `--src` redirects only the infer-split source, while the trained schema and preprocess state stay anchored to the run.

Outputs: canonical `predictions.jsonl`, a native mirror dataset under `dst/dataset` in the source's on-disk format (tabular: input columns plus `<field>_pred` columns; vision: native annotations re-readable by the same io), and optional visualizations and summary tables. The mirror is additive, and `predictions.jsonl` is never replaced. Round trip is supported for tabular (csv, parquet) and vision. Forecast or windowing tabular and writer-less formats fail fast on the mirror but still emit `predictions.jsonl`.

### Flow

Load saved config → replay overrides → rebuild the routed datamodule → `datamodule.bind_infer_source(--src)` → rebuild the module → `trainer.predict` over `datamodule.predict_dataloader()` (checkpoint loaded; `predict_step` returns per-batch dicts keyed by activation-node name, and the host merges `infer.ctx` with per-batch restore values from `predict_ctx_builder`, so decoded atoms land in original coordinates) → `CaptureMap` turns each batch into `capture.*` atoms → `datamodule.write_predictions` hands every writer node the keys its `in` names and writes the native mirror → emit `predictions.jsonl`, one line per sample holding `sample_id` plus every `capture.*` atom of that sample.

The source is the run directory only. Exported-bundle inference is out of scope. Prediction is a single-device `trainer.predict` for deterministic order, with the same orchestration, progress bar and logging as train and test.

## Constraints

- A writer node MUST read predictions only through `capture.*` keys, and a key it names that the `capture:` map does not produce fails fast at build.
- The infer runner MUST NOT branch per modality. The data graph is encoded entirely in the saved config.
- Native-mirror write-back MUST preserve source columns and row order and fail fast on a `<field>_pred` name collision.
- A datamodule lacking `write_predictions` MUST fail fast under `teia infer`.
- A run whose datamodule is stream-driven has no per-sample prediction dataset and no `predict_dataloader`, so `teia infer` fails fast on it.

