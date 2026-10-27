#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/tabular_multi_cls.py data/tabular-multi-cls

teia train \
  task=tabular-multi-cls \
  project=demo \
  experiment=tabular-multi-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/tabular-multi-cls \
  "$@"
