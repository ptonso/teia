# `one_to_one/grid/`

**Contract.** One input, one output. The input carries an axis with symmetric locality (adjacency usable by a sliding kernel) - a spatial grid, typically `[C, H, W]`.

**Why this is a real constraint.** A convolution assumes neighboring positions are meaningfully adjacent. Feed it a permuted or order-free axis and the inductive bias is meaningless, or wrong.

**Members**
- `timm.py` - `TimmEncoder`. timm backbone in pooled, spatial or tokens mode.

**Placement.** A component belongs here iff it declares exactly 1 `in`/`out` and its forward math requires spatial adjacency. If order (not locality) is the binding constraint, it is `sequence/`; if the axis is permutation-symmetric, it is `set/`. See `specs/delta/nn-taxonomy-reorg.md` §1.2–§1.4.
