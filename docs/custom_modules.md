# Custom modules

When no shipped preset fits — you have your own architecture, or a loss Teia doesn't provide — you
write the module in your own Python package and wire it in with config. Teia never needs to be
forked or modified; it resolves your classes by import path. The graph shape and the `batch.*` keys depend on the task contract.


## The module graph

A Teia model is a `TeiaNetModule`: a small graph of **nodes** plus the optimizer config. A
preset is just the YAML that describes that graph. Three common node roles:

- **Forward nodes** — your network (embedder, encoder, neck, head). They read tensors and write
  tensors.
- **Activation nodes** — own the prediction route: a traced tensor activation plus a host-side decode.
- **Loss nodes** — pure backward terms. They read predictions and targets, then write scalar
  `loss.*` tensors.

Nodes communicate through a typed **workspace** keyed by namespace:

| Prefix    | Source              | Readable in forward | Readable in loss |
|-----------|---------------------|:-------------------:|:----------------:|
| `batch.*` | the dataloader batch| ✓                   | ✓                |
| `feat.*`  | forward-node output | ✓                   | ✗ (forward-only) |
| `pred.*`  | forward-node output | ✓                   | ✓                |
| `act.*`   | activation output   | —                   | —                |
| `post.*`  | activation decode   | —                   | —                |
| `loss.*`  | loss-node output    | —                   | ✓ (written)      |

The `batch.*` keys are fixed by the task contract (`teia.task.*`). Pre-flight rejects a netmodule that reads a `batch.*` key its task does not declare.

Each node declares what it reads and writes with a flat `in`/`out` envelope, sitting right beside the
node's own constructor kwargs (no nesting):

```yaml
head:
  _target_: my_pkg.nn.MyHead
  in: batch.image              # str or list[str] — workspace keys the node reads
  out:
    pred.logits: [num_classes] # {key: shape | null} — keys the node writes; shape has no batch dim
  weight: 1.0                  # loss nodes only — weight in the multi-task sum
```

The rule that classifies a loss node is simple: **a node is a loss node iff any `out` key starts with
`loss.`**. Activation nodes conventionally write `act.*`. There is no `kind:` field. The reserved
envelope keys (`in`, `out`, `weight`, `optimizer`, `detach`, `aux_in`, `last_layer`, `bind`, …) are
stripped before your class is constructed — they're orchestration metadata, never constructor
arguments (so a constructor kwarg cannot reuse one of those names).

> The complete contract — reserved keys, validation rules, multi-output losses — is
> [core/module/user_contracts.md](../specs/core/module/user_contracts.md).

## The `TeiaNode` contract (forward nodes)

A forward node subclasses `TeiaNode` ([teia/base/net](../src/teia/base/net/node.py)). You
implement two methods:

```python
from teia.base.net import TeiaNode

class MyNode(TeiaNode):
    def build_module(self, **kwargs) -> None:
        # kwargs are exactly the YAML keys under this node.
        # self.in_shape / self.out_shape are INJECTED by the orchestrator,
        # as feature-dim tuples with NO batch dim — e.g. (3, 224, 224) or (256,).
        ...

    def forward(self, *args):     # positional args arrive in the node's `in` order
        ...
```

If a node needs dataset metadata it can't get from shapes — number of classes, vocab sizes — it can
implement the optional hook `configure_from_datamodule(self, dm)`; Teia calls it during build. (A
classification head can use it to read `dm.num_classes`.)

## The `BaseActivation` contract (prediction route)

An activation node subclasses `BaseActivation` ([teia/base/net](../src/teia/base/net/node.py))
and owns the whole **prediction route**: `forward → activation → kernel`.

```python
from teia.base.net import BaseActivation

class MyActivation(BaseActivation):
    def activation(self, *logits) -> dict[str, Tensor]:   # pure-tensor, ONNX-TRACEABLE → act.*
        ...

    @staticmethod
    def kernel(activated, ctx) -> list[dict]:             # host-side decode (NMS, argmax, …) → post.<node>.*
        ...
```

- `activation` must be pure tensor ops — it is traced into the exported ONNX graph.
- `kernel` runs on the host and returns one flat `{atom: value}` dict per sample. It is copied
  verbatim into the export bundle, so it must be defined inline and read only `activated`, `ctx` and
  literals from `params()`. A route whose activated tensors already are the prediction sets
  `RAW_PASSTHROUGH = True` instead.
- Don't override `postprocess`; it is the plumbing that calls your kernel.

## The `capture:` map (what leaves the net)

A netmodule ends with a `capture:` map from the task's capture atoms to `act.*` or `post.*` keys:

```yaml
capture:
  cls.scores: act.probs        # a traced tensor
  cls.label: post.act.label    # an atom your kernel returned (node `act`, atom `label`)
```

These atoms are what the task's evalmodule reads, and what infer's writers turn into prediction
files. With `task=` set, pre-flight checks that every atom the contract requires is mapped. Without
a task you may capture any atoms you like, as long as your evalmodule reads the same names.

## The `BaseLoss` contract (loss nodes)

A loss node subclasses `BaseLoss` ([teia/base/net](../src/teia/base/net/node.py)). It is a
pure backward term: `forward(*in_keys) -> Tensor`.

```python
from teia.base.net import BaseLoss

class MyLoss(BaseLoss):
    def forward(self, *in_keys) -> Tensor:
        ...
```

Loss nodes need no special hook to reach the forward node — everything arrives through positional
`in` arguments, in declared order.

> The prediction route is specified in [core/module/activation_route.md](../specs/core/module/activation_route.md).

## Wiring it in: `_target_` and external packages

Every node config names its class with `_target_`, a dotted import path, plus its constructor
kwargs:

```yaml
head:
  _target_: my_pkg.nn.MyHead       # resolved by a normal Python import
  hidden: 96                        # constructor kwarg (matches build_module signature)
  in: batch.img
  out:
    pred.logits: [10]
```

`_target_` resolves through `import`, so your package only needs to be importable — install it
editable (`pip install -e .`) and reference `my_pkg.nn.MyHead`. Keep your node/preset YAML under a
project `conf/` overlay (see [Configuration](configuration.md)) and select it with the
group-override syntax.

---

Specs for the precise contract:
[user_contracts.md](../specs/core/module/user_contracts.md) ·
[activation_route.md](../specs/core/module/activation_route.md) ·
[nodes.md](../specs/core/module/nodes.md) ·
[batch.md](../specs/base/batch.md)
