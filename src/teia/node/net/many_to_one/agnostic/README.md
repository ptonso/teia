# `many_to_one/agnostic/`

**Contract.** N inputs, 1 output. Every input is a fixed-shape tensor with no shared structural requirement beyond that.

**Why this is a real constraint.** Fusing multiple fixed-shape vectors (concat / sum / gate) works regardless of what any of them represents. The only requirement is compatible tensor shape.

**Members**
- `concat.py`: `ConcatFusion`. Concatenates N inputs along an axis, with optional flatten to `[B, D]`.
- `sum.py`: `SumFusion`. Element-wise sum of N same-shape inputs.
- `gated.py`: `GatedFusion`. Learned softmax-gated weighted sum of N same-shape inputs.

**Placement.** A component belongs here iff it declares at least 2 `in` and exactly 1 `out`, and every input is an unstructured fixed-shape tensor. If any input carries a real structural requirement (order, locality, permutation-symmetry), or the inputs disagree, see `set/`, `sequence/`, `grid/` or `heterogeneous/`.
