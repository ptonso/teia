# teia specs

`teia` is a Hydra-first training runtime. It composes a configuration into `train`, `test`, `infer`, `export` and `eval` stages over one Lightning module. Three config-declared component graphs do the work: an **data graph** turns raw data into typed `batch.*` fields, an **net graph** turns those into predictions and losses, and an **eval graph** turns a captured run into metrics and plots. Structure (`set`, `sequence`, `grid`, `graph`) is the one axis that organizes both data and compute, and there are no modality packages.

These specs describe the contracts a contributor must respect.

## Reading order

1. [architecture](architecture.md): the parts of the system, the four axes, boundaries, dependency direction, shared terms.
2. [terminology](terminology.md): why domain words never name code, and the triage for finding them.
3. [_template](_template.md): how a spec is written here.
4. [stability](stability.md): the three layers, how each one changes, and what a version number promises.
5. [ecosystem](ecosystem.md): plugins, where each kind of code lives, reuse, baselines, licenses and publishing.
6. [plugin](plugin.md): what a plugin package must contain, and the rules `teia plugin check` verifies.


**Contracts (`base/`).** The torch-only floor shared by every component.

1. [base/structure](base/structure.md): the structure axis and value types.
2. [base/batch](base/batch.md): the field vocabulary at the data↔net seam.
3. [base/data/nodes](base/data/nodes.md): the six data node ABCs, phases, stages and kernels.
4. [base/net/nodes](base/net/nodes.md): `TeiaNode` and the build hooks.
5. [base/net/activation](base/net/activation.md): activation and loss bases.
6. [base/eval/nodes](base/eval/nodes.md): metric, view and comparison ABCs.
7. [base/task](base/task.md): the task contract over the two graph boundaries.

**Core (`core/`).** The engine that executes the graphs.

1. [core/overview](core/overview.md): the package surface and workflow.
2. [core/config](core/config.md), [core/task](core/task.md), [core/cli](core/cli.md), [core/deps](core/deps.md): configuration, tasks and their presets, commands and dependencies.
3. [core/runtime](core/runtime.md): stage execution, run layout and guards.
4. [core/data_graph](core/data_graph.md), [core/datamodule](core/datamodule.md): the data plane and its executor.
5. [core/module/overview](core/module/overview.md) and its siblings: the net plane.
   - [user_contracts](core/module/user_contracts.md), [nodes](core/module/nodes.md), [objective](core/module/objective.md), [activation_route](core/module/activation_route.md), [capture_map](core/module/capture_map.md)
   - [codegen](core/module/codegen.md), [dry_run](core/module/dry_run.md), [bind](core/module/bind.md), [optimization](core/module/optimization.md)
6. [core/eval](core/eval.md), [eval/layout](eval/layout.md): capture and the eval plane, and the layout of its components.
7. [core/infer](core/infer.md), [core/export](core/export.md): prediction and standalone bundles.

**Component libraries.** Where a component lives and what it must obey.

1. [data/layout](data/layout.md): the six data-node kind folders and the authoring patterns.
2. [net/layout](net/layout.md), [net/types](net/types.md): `fan × assumes` placement, and the types across a forward boundary and the head → activation → loss route.
3. [docstrings](docstrings.md): the provenance docstring every component carries.
