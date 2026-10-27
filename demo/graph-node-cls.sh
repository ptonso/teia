#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/graph_node_cls.py data/graph-node-cls

teia train \
  task=graph-node-cls \
  project=demo \
  experiment=graph-node-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/graph-node-cls \
  "$@"
