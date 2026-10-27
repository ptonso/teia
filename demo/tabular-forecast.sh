#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/tabular_forecast.py data/tabular-forecast

teia train \
  task=tabular-forecast \
  project=demo \
  experiment=tabular-forecast \
  epochs=${EPOCHS:-100} \
  data_root=data/tabular-forecast \
  "$@"
