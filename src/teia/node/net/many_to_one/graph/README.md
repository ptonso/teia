# `many_to_one/graph/`

**Contract.** N inputs, 1 output. One input is a padded batch of node features `[B, N, D]`, and the inputs carry an explicit dense adjacency `[B, N, N]` (1 where two nodes are connected, self-loops included) plus a node mask `[B, N]` (1 for a real node, 0 for padding). The output is `[B, N, D]`.

**Why this is a real constraint.** A message-passing or graph-attention step is only defined given an explicit edge structure, and padding makes the mask part of the input: without it a padded slot would take part in attention, normalisation and pooling. `N` is the largest board of the batch, so it changes from batch to batch and no weight may depend on it.

**Members**
- `graph_encoder.py` - `GraphEncoder`. Message-passing layers in Graphormer's AGGREGATE / COMBINE form, both chosen by name. Its leaves reproduce Graphormer, GATv2, GraphSAGE and GIN.
- `aggregate.py`, `combine.py` - the AGGREGATE and COMBINE options `GraphEncoder` selects by name. Building blocks, not nodes.

**Placement.** A component belongs here iff it declares ≥2 `in`, one of them an explicit adjacency, and exactly 1 `out`. A per-node op that ignores the adjacency (a shared MLP head, a per-node projection) is `agnostic/`, even when its input happens to be node features.
