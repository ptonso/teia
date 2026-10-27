#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/tabular_survival.py data/tabular-survival

teia train \
  task=tabular-survival \
  project=demo \
  experiment=tabular-survival \
  epochs=${EPOCHS:-100} \
  data_root=data/tabular-survival \
  "$@"
