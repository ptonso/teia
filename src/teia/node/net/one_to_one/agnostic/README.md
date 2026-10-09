# `one_to_one/agnostic/`

**Contract.** One input, one output. The input is any fixed-shape tensor — no requirement on its
structure or value domain.

**Why this is a real constraint.** Any op that works identically regardless of what the input
tensor represents belongs here. If it needs a stronger guarantee, it belongs elsewhere.

**Members**
- `flatten.py` — `Flatten`. Parameter-free `[B, *dims] -> [B, prod(dims)]` shape adapter.
- `mlp.py` — `MLP`. Generic dense feed-forward stack: `Linear -> [activation -> Linear] x N`. Reused
  as-is wherever a dense stack is needed — not duplicated.
- `linear.py` — `LinearHead`. Single affine projection to a declared output shape.
- `pointwise_mlp.py` — `PointwiseMLP`. Dense feed-forward applied independently to every row.

**Placement.** A component belongs here iff it declares exactly 1 `in`/`out` and places no
