# Changelog

Versioning rules: [specs/stability](specs/stability.md).

## 0.2.0 - 2026-10-27

Teia ships machinery and reference leaves. Tuned wiring and system recipes belong to the projects that tune them.

### Breaking

- Interactive (reinforcement-learning) sources are engines. The `BaseRlIO` environment contract, `EnvReader` (`reader/env_reader`) and the runners' hand-written `item.*` records are removed. An engine ([specs/base/data/nodes](specs/base/data/nodes.md#engine)) turns actions into named signals and never resets inside `step`, the runner resets finished environments from `runner.boundary`, and `reset_policy` and the validity item move from the engine to the runner. Simulator backends become `Engine` subclasses.
- Stream buffers (`ReplayBuffer`, `PrioritizedReplayBuffer`, `SequenceBuffer`, `RolloutBuffer`) bind their roles through a mapping-form `in` (`{reward: item.reward, terminated: ..., truncated: ...}`) and raise on a missing or unknown role.
- `EpisodeAggregate` reads the runner-owned episode records `episodes.reward`, `episodes.length` and `episodes.ended_by`, and `from_source` requires `length` and `ended_by`.
- Task contracts register the batch fields they declare and declare `feedback` and `engine_atom`. The interactive fields (`obs_state`, `action`, `next_obs_state`, `log_prob`, ...) leave `FIELD_VOCAB`, so a collate may emit them only under a contract that declares them.
- `act_forward` runs only the netmodule nodes the `capture` map needs, and a capture map may name `pred.*` sources, which it previously rejected.
- `NormalizeImage` is replaced by the generic `Normalize` transform (fixed, fit and running statistics, `clip`, any `axis`). The `normalize*` leaves target it.
- A plugin that ships a task preset or a reference leaf at a path `teia` now owns (`task/rl-discrete`, `task/rl-continuous`, `task/rl-continuous-replay`, `node/data/engine/gymnasium` and the new leaves below) fails on composition, since a config path must be unique across installed packages. Drop the copy or give it the plugin's prefix, and select the `teia` preset with explicit overrides for the plugin's own netmodule, datamodule, callbacks and export.
- Removed netmodules `vision-cls/mlp`, `vision-cls/wide_mlp` and `tabular-cls/numerical_mlp`. Each task runs a `linear_probe` netmodule instead, and a project writes its own architecture.
- Removed `graph-node-cls/{coco_graph,coco_knn,labelme_graph}` datamodules and `graph-node-cls/timm_{gatv2,graphformer,no_graph}` netmodules.
- Removed the `vision_default` trainer, the `vision_default` and `vision_fitness` callbacks and the `adamw_graph` optimizer. `adamw_warmup_decay` stays.
- Evalmodules are folded by capture family instead of by task. `evalmodule/<task>/default` becomes `evalmodule/<family>/default`: `vision-cls`, `audio-cls`, `audio-wave-cls`, `tabular-cls` and `graph-node-cls` are `cls/default` (their files were identical), `vision-multi-cls`, `audio-multi-cls` and `tabular-multi-cls` are `multi-cls/default`, and `vision-det`, `vision-obb`, `vision-pose`, `vision-inst-seg`, `vision-sem-seg`, `vision-gen` (with `gfid`), `tabular-reg`, `tabular-survival` and `tabular-forecast` become `det`, `obb`, `pose`, `inst-seg`, `sem-seg`, `gen`, `reg`, `survival` and `forecast`. Presets and `evalmodule=` selections change accordingly.
- Removed the head leaf `linear_cls`, identical to `linear`. Select `node/net/head@...=linear`.
- Removed the board data nodes `CocoGraph`, `LabelmeGraph`, `CropNodes` and `NodeBoxes` with their leaves.
- `GraphNodeCls` is typed on node features: `batch.node_box` and `batch.node_image` are replaced by `batch.node_feat [B N F]`. The task reads TU Dortmund graph files.
- `PadNodeBoxes` is renamed `PadNodeRows` (leaf `pad_node_rows`, field `node_feat`), and `EdgeAdjacency` reads the length of any per-node tensor as its second input.
- `NodePack(box_dim=...)` is renamed `NodePack(attr_dim=...)`.
- `evalmodule/gen/gfid` is removed. Generation FID needs samples from a prior and a reference file, which Teia does not ship. It moves out of Teia into a research project. `FrechetDistance(reference=...)` stays.
- Demo scripts are renamed `demo/vision-cls.sh` and `demo/tabular-cls.sh`, and their data folders are `demo/data/{vision-cls,tabular-cls}`. The `prepare_*.py` scripts move to `demo/prepare/` without the prefix.

### Added

- Engine and command data nodes and the stream-mode datamodule: an engine feeds role-bound buffers through one collect-and-act cycle (`SyncRunner`, `AsyncRunner`), a command node holds per-environment goals redrawn on episode resets, and the actor runs only the capture slice. The runner owns the episode records and the evaluation graph reads them as `episodes.*`.
- `data_root` takes a registered world as `<scheme>:<id>` (`gym:CartPole-v1`) beside a path.
- `LinearHead(init_gain=...)`, the run directory reaching the datamodule, and a log path under it for each engine.
- `linear_probe` netmodules (`vision-cls`, `tabular-cls`, `graph-node-cls`) on the shared fragment `netmodule/_shared/linear_probe`.
- Task wiring (preset, `csv` datamodule, `linear_probe`, demo) for `tabular-multi-cls` and `tabular-reg`.
- Task wiring (preset, datamodule, `linear_probe`, demo) for `tabular-survival` (`csv`) and `tabular-forecast` (`window`). The window datamodule does not feed the target column as an input, so history reaches it through a lag column.
- Task wiring (preset, datamodule, `linear_probe`, demo) for `vision-det`, `vision-obb`, `vision-pose` and `vision-inst-seg`. The probe is a grid of detection candidates read from the flattened image: `GridDetLoss` assigns each box to the cell holding its center (`extra: none | angle | keypoints`), `GridDetActivation` decodes the grid (`extra: none | angle | keypoints | polygon`, the polygon being the box rectangle), and the shared postprocessing (`grid_postprocess`, `greedy_nms`, `restore_xy`, `frac_cxcywh`) lives in `node/net/_activation_utils`. The datamodules (`yolo_format` for det, pose and inst-seg, `labelme` for obb) apply no augmentation, scale pixels to `[0, 1]` and use batch 64. The demos draw shapes with `demo/prepare/shapes.py`. The pose demo scores near zero AP, because the OKS tolerance is under one pixel at 32 px, while its keypoint error is less than half that of a probe that learned nothing.
- Task wiring (preset, `image_folder_unlabeled` datamodule, `linear_probe`, demo) for `vision-gen`, a linear autoencoder with a 32-dim latent scored by rFID. Objectives: `ReconLoss` (pixel error, with `sigma: fixed | optimal | optimal_per_image` for a calibrated Gaussian decoder), `ReconstructActivation` and the shared `_pixel_space` helpers, with tests. The `frechet_inception` leaf resolves `device: auto` (cuda when available, else cpu), so rFID runs without a GPU.
- Objectives for them: `CoxActivation` and `CoxLoss` (leaves `act/cox`, `loss/cox`; ties share one risk set, Breslow) and `QuantileActivation` and `QuantileLoss` (leaves `act/quantile`, `loss/quantile`), with tests.
- Task wiring (preset, format datamodule, `linear_probe`, demo) for `vision-multi-cls` (`image_folder`), `vision-sem-seg` (`mask_dir`), `audio-cls`, `audio-multi-cls` and `audio-wave-cls` (`audio_folder`). The datamodules apply no augmentation. The two mel-spectrogram probes set `lr: 1.0e-4`, since the log-mel grid is unnormalized.
- `TuReader` (leaf `reader/tu`, with an `extension` argument like `CocoReader`) and the `graph-node-cls/tu` datamodule, for TU Dortmund graph datasets. Node labels may be names as well as integers, and `edges` selects the edge file `<DS>_<edges>.txt` (default `A`), so several graphs over the same nodes share one folder. The dataset prefix comes from the single `*_graph_indicator.txt`.
- Fused task `vision-graph-node-cls` (contract `VisionGraphNodeCls`, `graph-node-cls` with an image per node), with its preset, the `tu_node_images` datamodule (TU files plus `nodes/<id>.png`), a `linear_probe` (flattened node image packed with the node features, then a pointwise linear head) and a demo. `NodeImages` (leaf `reader/extension/node_images`) and `DecodeNodeImages` (leaf `transform/decode_node_images`, `size: null` keeps the stored pixels) come with their tests.
- `callbacks/fitness`: the core callback set with the checkpoint and early stopping monitoring the reserved `val/fitness` key.
- Task `rl-discrete` (contract `RlDiscrete` with the shared field sets `_rl_fields`, preset, `gymnasium` datamodule, `linear_probe`, demo and the `rl` evaluation family). The world is `data_root=gym:<id>`, served by `GymnasiumEngine` and its `VectorEngine` base (extra `pyteia[rl]`). The probe is a linear Q-function over `QArgmaxActivation` (epsilon-greedy, with a `LinearSchedule` counted in environment steps) and `DqnLoss` (Bellman regression on a frozen, hard-synced copy; Huber by default), with tests against the textbook definitions. The reference leaves `reshape/replay`, `act/q_argmax`, `loss/dqn` and `engine/gymnasium` come with it, and `callbacks/online` checkpoints on the episodic return.
- Tasks `rl-continuous-replay` and `rl-continuous` (contracts `RlContinuousReplay` and `RlContinuous`, presets, `gymnasium` datamodules on Pendulum, `linear_probe`, demos), on the `rl` evaluation family. The replay probe is a linear soft actor-critic: `TanhGaussianBoxActivation`, `SquashedGaussianSample`, twin `QCritic` with EMA targets, a learned temperature (`Parameter`) and `SacCriticLoss`, `SacActorLoss` and `SacAlphaLoss` on three optimizers. The on-policy probe is a linear PPO: `GaussianBoxActivation`, a state-independent `Parameter` log standard deviation, `PpoPolicyLoss` and `ValueLoss`, over the `rollout` reshape leaf. Leaves: `act/{tanh_gaussian_box,squashed_gaussian_sample,gaussian_box}`, `head/q_critic`, `branch/parameter`, `loss/{sac_critic,sac_actor,sac_alpha,ppo_policy,value}`. Tests check the losses and the squashed log-probability against the textbook definitions. `SacAlphaLoss` raises without a target entropy or a datamodule `action_dim` instead of assuming `-1`.
- `demo/graph-node-cls.sh` with a synthetic TU generator, and the `demo` end-to-end pack (`tests/demo-runs/run-all.sh`).
- `tests/test_task_conformance.py`: every shipped task preset and module passes its contract, and Teia ships no netmodule other than the probes.

### Results may change

- `task=vision-cls` and `task=tabular-cls` train a linear probe instead of an MLP.

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
