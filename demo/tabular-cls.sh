#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/tabular_cls.py data/tabular-cls

teia train \
  task=tabular-cls \
  project=demo \
  experiment=tabular-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/tabular-cls \
  "$@"
