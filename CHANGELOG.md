# Changelog

Versioning rules: [specs/stability](specs/stability.md).

## Unreleased (0.1.0)

### Breaking

- Task contracts `RlDiscrete`, `RlContinuous`, `RlContinuousReplay`: the transition fields `reward`, `terminated`, `truncated` and `episode_id` are now target fields, captured as eval ground truth.
- `evalmodule.eval_episodes` is removed and raises as an unknown setting. The eval episode count is `datamodule.env.eval_episodes`.
- `TeiaDataModule.evaluate` and `InteractiveRegime.evaluate` are removed. Eval episodes run through the capture-only test dataloader.

### Added

- `EpisodeAggregate` metric (`node/eval/metric/episode_aggregate`): per-episode `episode_return`, `episode_length` and `terminated_rate` from captured per-step rows.

### Results may change

- Interactive eval episodes: episode ids are unique across envs, and the autoreset step after an episode ends is no longer yielded.
- Interactive eval episodes run the item nodes with `training` off, so `PolicyAction` acts greedily and online-fit transforms stay frozen. Eval previously explored with the policy's training-time sampling (epsilon-greedy from the schedule start for `q_argmax`).

## 0.0.1

First release.
