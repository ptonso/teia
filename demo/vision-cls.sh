#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/vision_cls.py data/vision-cls

teia train \
  task=vision-cls \
  project=demo \
  experiment=vision-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/vision-cls \
  "$@"
