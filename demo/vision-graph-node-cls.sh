#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/vision_graph_node_cls.py data/vision-graph-node-cls

teia train \
  task=vision-graph-node-cls \
  project=demo \
  experiment=vision-graph-node-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/vision-graph-node-cls \
  "$@"
