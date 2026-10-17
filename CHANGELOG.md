# Changelog

Versioning rules: [specs/stability](specs/stability.md).

## 0.1.0 - 2026-10-17

### Breaking

- Task contracts `RlDiscrete`, `RlContinuous`, `RlContinuousReplay`: the transition fields `reward`, `terminated`, `truncated` and `episode_id` are now target fields, captured as eval ground truth.
- `evalmodule.eval_episodes` is removed and raises as an unknown setting. The eval episode count is `datamodule.env.eval_episodes`.
- `TeiaDataModule.evaluate` and `InteractiveRegime.evaluate` are removed. Eval episodes run through the capture-only test dataloader.
- The capture encoder raises on a float image blob outside `[0, 1]` instead of guessing a signed range. Kernels and transforms that emit blobs convert to unit range; a `pixel_norm: signed` generation run now fails at capture on its `batch.image` target.
- `summary` raises when two inputs publish one scalar key with different values; `namespace: true` writes `<input atom>.<field>`.
- `evalmodule/vision-gen/default` scores `rfid` (Inception pool3, reconstructions vs test images) and writes `rfid.*` summary keys; `frechet_distance`/`kernel_distance` keys of the old profile are gone.
- `FrechetDistance` and `KernelDistance` raise on an empty side instead of returning counts only. `KernelDistance` averages a subset MMD^2 with `kernel: poly3` (KID) by default; `rbf` is the previous kernel.
- `EmbeddingAccumulator.update_paths` takes its batch size from the constructor.
- The `seed_aggregate` compare node is removed. `runtime.n_runs` with `teia eval --run-dir <group>` reports mean and sample std (ddof=1) per replica set.

### Added

- `teia sweep`: Hydra sweep expressions in the overrides define a grid or seeded random search, scored on `sweep.objective` (`sweep.n_trials`, `sweep.mode`) over `runtime.n_runs` replicas without computing test.
- `runtime.n_runs`: `teia train` trains the config n times into a group directory of `rep_i` runs with seeds `runtime.seed + i`, then writes an aggregate report.
- Cross-replica reducer (`teia.core.eval.reduce`): reduces metric results by value shape into the mean with `_std` and `_values` siblings; `ranked_bar` draws error bars and `multi_curve_bundle` draws `y_std` bands.
- Task `graph-node-cls` with its `graph-node-cls/{coco_graph,coco_knn,labelme_graph}` datamodules and `timm_{gatv2,graphformer,no_graph}` netmodules.
- `GraphEncoder` (`graph_{gatv2,gin,graphormer,sage,transformer}`) and the `NodePack`/`NodeUnpack` adapters.
- Graph data nodes: `EdgeAdjacency`/`KnnAdjacency`, `CropNodes`, `NodeBoxes`, the `coco_graph`/`labelme_graph` reader extensions and the graph collates.
- `adamw_warmup_decay` and `adamw_graph` optimizer leaves, the `vision_default` and `vision_fitness` callbacks and the `vision_default` trainer.
- `BalancedCrossEntropyLoss` (`balanced_cross_entropy`).
- `TimmEncoder` (`timm_resnet18`, `timm_siglip`; extra `vision`) with `chunk_size`: checkpointed chunks in training, plain chunks in eval and when frozen.
- `EpisodeAggregate` metric (`node/eval/metric/episode_aggregate`): per-episode `episode_return`, `episode_length` and `terminated_rate` from captured per-step rows.
- `test.weights` (`best` or `last`): which checkpoint test and export load when none is given.
- `InceptionPool3` FID feature extractor (`node/eval/metric/unpaired/inception`, extra `fid`): pytorch-fid pool3 with sha256-pinned weights.
- `FrechetDistance(reference=...)` with `reference_statistics`/`save_reference`: precomputed real-set statistics.
- `evalmodule/vision-gen/gfid`: generation FID against `${data_root}/fid/inception_pool3_train.npz`.
- `evalmodule.media` caps now apply, to gif and mp4 blobs only.

### Results may change

- Interactive eval episodes: episode ids are unique across envs, and the autoreset step after an episode ends is no longer yielded.
- Interactive eval episodes run the item nodes with `training` off, so `PolicyAction` acts greedily and online-fit transforms stay frozen. Eval previously explored with the policy's training-time sampling (epsilon-greedy from the schedule start for `q_argmax`).

## 0.0.1

First release.
