# Run baselines

A research project compares its method with existing ones. The comparison is fair when every method is scored on the same split by the same evaluation, and Teia's evalmodule guarantees that for anything that produces the task's capture atoms. What remains to decide for each baseline is how much else to share, and where its code lives. Most baselines deserve little effort; the one or two closest to the user's method deserve the most.

## Choose how each baseline runs

Pick the cheapest way that keeps equal whatever the user's claim is about. Ask the user what the claim is when it is unclear.

| Way | When | How |
|---|---|---|
| **Trained** | the claim is about a component, so only that component may differ: same data pipeline, same training loop | write or reuse the baseline's components and a netmodule in the task's folder, then train it like the user's method ([write-config](write-config.md), [write-component](write-component.md)) |
| **Frozen** | a pretrained model, an external pipeline or an API is used as released | one forward node that calls the external system and returns what the task's capture atoms need, in a netmodule with no loss node; run it with `epochs=0`, which skips training and still runs the test pass and the evaluation. The node needs no trainable parameter: with none, Teia builds no optimizer |
| **Imported** | the baseline only runs in its own framework | Teia has no importer for saved predictions yet: write a frozen node that looks up each sample's saved prediction by a key the datamodule provides (inspect the batch fields), or ask the user before building more |

Many baselines of one family (twenty detectors, every backbone of one library) need one adapter component whose constructor takes the model name, and one leaf or module per model.

## Where the code lives

1. **Look before writing.** Search the installed packages for a component that already wraps the same upstream ([inspect](inspect.md#does-it-already-exist)). If one exists, depend on its package.
2. **A wrapper used by this project** lives in this project's plugin, under `src/<package>/node/`, with its configs in the owner folders ([make-a-plugin](make-a-plugin.md#layout)).
3. **A wrapper several projects need** belongs in its own small plugin, named after the library it wraps. Propose this to the user; do not create it on your own.
4. **The upstream code itself** is installed as a pinned dependency: a released version, or `name @ git+<repository>@<commit>` in `pyproject.toml`. Copy upstream source into the project only when it cannot be installed and its license allows copying, and keep it unmodified under `third_party/<name>/` with its license file.
5. **Copyleft code** (GPL, AGPL) imposes its license on whatever includes it. Tell the user before wrapping it, because it decides the license of the plugin.

A model the user trained earlier is their own artifact: its docstring says in `Description` how it was produced. A component that runs code or weights from another repository adds an `Upstream` block after `Source`:

```
Upstream:
  repo: "<repository URL>"
  commit: "<commit hash or tag>"
  license: "<SPDX identifier>"
```

## Show the baseline runs correctly

| Way | Check |
|---|---|
| Trained | load the original weights into the wrapped model and compare its outputs with the original code on the same inputs; for the baselines closest to the user's method, also reproduce the original paper's number under its own settings |
| Frozen | run the original's own example input through the wrapper and compare with the original's output |
| Imported | none: the predictions come from the original code |

Keep each check as a test or a script in the project, and its result in the project's notes, so the comparison can be defended later.

## Report

For each baseline: the way it runs and why, where its code lives, its upstream repository, commit and license, the check that was run and its result, and the command that produces its row of the comparison.
