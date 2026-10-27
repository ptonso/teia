# `zero_to_one/`

**Contract.** No declared input, one output. There is no `assumes` subfolder here - with zero inputs there is nothing to constrain.

**Why this is a real constraint.** A node with no `in_key` cannot depend on shape or value requirements of an input that does not exist. Any positional argument passed is a dummy used only to fix graph ordering, never real data.

**Members**
- `parameter.py`: `Parameter`. A trainable tensor of the declared out shape, emitted at every forward (a state-independent log standard deviation, a log-temperature).

**Placement.** A component belongs here iff it declares zero `in` keys. If it later grows a real input, it moves to `one_to_one/` (or beyond) and picks up an `assumes` subfolder.
