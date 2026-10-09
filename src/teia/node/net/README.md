# `net/`

Components are organized by **fan × assumes**:

- **fan** — how many inputs and outputs a node declares.
- **assumes** — what must be true of its input(s) for the forward math to be defined.

**Placement rule:** a folder is an interchangeability class. Two components in the same
`<fan>/<assumes>/` folder must be swappable — change the `_target_` and the graph still runs. If a
distinction doesn't prevent a swap, it's a file, not a folder.

## `fan`

Counted from the length of the node's declared `in`/`out` lists in conf.

| fan | in-arity | out-arity |
|---|---|---|
| `zero_to_one` | 0 | 1 |
| `one_to_one` | 1 | 1 |
| `many_to_one` | ≥2 | 1 |
| `one_to_many` | 1 | ≥2 |
| `many_to_many` | ≥2 | ≥2 |

`zero_to_one` takes no `assumes` subfolder — with zero inputs there is nothing to assume.

## `assumes`

Precedence order, first match wins:

1. **value-domain** (`categorical` / `count` / `numerical`) — only on `one_to_one` raw-column
   tokenizers; breaks with `IndexError`/`NaN` if violated.
2. **`sequence`** — an axis carries binding order (recurrence, causal masking).
3. **`grid`** — an axis carries symmetric locality, consumed by a sliding kernel.
4. **`set`** — a variable-length, permutation-symmetric axis with shared weights.
5. **`agnostic`** — any fixed-shape tensor; no requirement.
6. **`graph`** — explicit adjacency (reserved; zero members today).
7. **`heterogeneous`** *(multi-input only)* — inputs share no single uniform requirement.

For a multi-input node, apply the ladder to every input. All agree → that token. They disagree →
`heterogeneous`.

## Placement procedure

1. Has a `forward` over shape-typed tensors (a `TeiaNode`)? If not → `activation/` or `loss/` (both
   flat, out of scope here).
2. Count declared `in`/`out` entries → **fan**.
3. If fan is `zero_to_one`, stop — no `assumes` subfolder.
4. Run the `assumes` ladder over every input.

