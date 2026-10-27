#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/tabular_reg.py data/tabular-reg

teia train \
  task=tabular-reg \
  project=demo \
  experiment=tabular-reg \
  epochs=${EPOCHS:-100} \
  data_root=data/tabular-reg \
  "$@"
