#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/shapes.py data/vision-inst-seg --format seg

teia train \
  task=vision-inst-seg \
  project=demo \
  experiment=vision-inst-seg \
  epochs=${EPOCHS:-30} \
  data_root=data/vision-inst-seg \
  "$@"
