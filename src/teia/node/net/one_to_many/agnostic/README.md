# `one_to_many/agnostic/`

**Contract.** One input, N outputs. The input is any fixed-shape tensor — no requirement beyond
that.

**Why this is a real constraint.** Splitting a fixed-shape feature into several downstream taps
works regardless of what the input represents structurally.

**Members**
- `split.py` — `Split`. Fans a single input out to N identical outputs — the generic multi-task
  tap used when several downstream heads each need their own copy of the same upstream feature.

**Placement.** A component belongs here iff it declares exactly 1 `in` and ≥2 `out`, and the input
places no structural requirement. If the input instead requires a spatial grid, it is
`one_to_many/grid/`.
