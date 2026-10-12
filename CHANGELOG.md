# Changelog

Versioning rules: [specs/stability](specs/stability.md).

## Unreleased (0.1.0)

### Breaking

- Task contracts `RlDiscrete`, `RlContinuous`, `RlContinuousReplay`: the transition fields `reward`, `terminated`, `truncated` and `episode_id` are now target fields, captured as eval ground truth.
- `evalmodule.eval_episodes` is removed and raises as an unknown setting. The eval episode count is `datamodule.env.eval_episodes`.
- `TeiaDataModule.evaluate` and `InteractiveRegime.evaluate` are removed. Eval episodes run through the capture-only test dataloader.
- The capture encoder raises on a float image blob outside `[0, 1]` instead of guessing a signed range. Kernels and transforms that emit blobs convert to unit range; a `pixel_norm: signed` generation run now fails at capture on its `batch.image` target.

### Added

- `EpisodeAggregate` metric (`node/eval/metric/episode_aggregate`): per-episode `episode_return`, `episode_length` and `terminated_rate` from captured per-step rows.
- `test.weights` (`best` or `last`): which checkpoint test and export load when none is given.
- `evalmodule.media` caps now apply, to gif and mp4 blobs only.

### Results may change

- Interactive eval episodes: episode ids are unique across envs, and the autoreset step after an episode ends is no longer yielded.
- Interactive eval episodes run the item nodes with `training` off, so `PolicyAction` acts greedily and online-fit transforms stay frozen. Eval previously explored with the policy's training-time sampling (epsilon-greedy from the schedule start for `q_argmax`).

## 0.0.1

First release.
