# Start from a task

This is the shortest road in Teia. A task carries a datamodule, a network, an evaluation and tuned run settings that were chosen to work together, so a run needs little beyond the task's name and the location of the data:

```bash
teia train task=<task> data_root=<dir>
```

`data_root` is the one dataset knob. Teia rejects `data=` and paths given as module selections, so that the dataset path is never confused with a config choice. Most of your work at this level happens before the command: choosing the right task and getting the data into the shape it expects.

## 1. Choose the task

Ask the user what they have and what they want out of it: what an input looks like, and what a prediction should be. Then list the tasks ([inspect](inspect.md)) and open the contracts of the plausible ones. A contract states the inputs it consumes and the predictions it promises, so the match between the user's goal and a contract is a matter of reading, and a wrong guess from the task's name is avoidable. If two tasks fit, say so and let the user decide.

If no task fits, this is a level 3 job. First check whether a task is close enough that its contract could be extended ([write-config](write-config.md#a-new-task)).

## 2. Get the data into shape

Every datamodule starts with a reader that understands one data layout, such as a folder per class, a manifest file or a table with named columns. The layout is defined by the reader, so ask the reader instead of guessing. Follow the trail:

1. `teia config show task=<task>` names the default datamodule.
2. `teia config show datamodule=<task>/<name>` lists the leaves it mounts. The first ones are the readers, and the transforms and collates after them show how raw fields become `batch.*`.
3. `teia config show node/data/reader=<leaf>` gives the reader's `_target_`. Open its source ([inspect](inspect.md#reading-what-comes-back)): the module docstring describes the layout it scans, how splits and labels are derived, and which file types it accepts.

Compare that description with what the user has. There are usually a few ways forward, and it is worth putting them to the user when they differ in cost:

- **Reshape the data** to the layout the reader expects. This is a small script in the project (rename, symlink, convert, write a manifest). Leave the original files untouched and write the prepared copy elsewhere, since you cannot know what else depends on them.
- **Choose another datamodule** in the task's folder, if one reads the user's format already ([swap](swap.md)).
- **Write a reader** for the format, when neither works and reshaping is unreasonable. That is [write-component](write-component.md).

Splits (train, validation, test) and label names usually come from the layout itself, and `meta.*` records what the datamodule learned. If a task needs a split that the data lacks, the reader's docstring says what it does about it.

## 3. Run, and read the first result

Try the real pre-flight and one batch before a long run ([validate in layers](SKILL.md#ground-rules)):

```bash
teia train task=<task> data_root=<dir> ++trainer.fast_dev_run=true
```

Data problems show up here as reader errors that name the path or column at fault. Then run for real ([run](run.md)) and read `eval/summary.yaml` in the run directory for the headline numbers. Report the run directory, the numbers, and the choices the task made for the user (module names from `teia config show task=<task>`), since those are what the next level of work would change.
