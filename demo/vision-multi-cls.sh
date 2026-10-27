#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/vision_multi_cls.py data/vision-multi-cls

teia train \
  task=vision-multi-cls \
  project=demo \
  experiment=vision-multi-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/vision-multi-cls \
  "$@"
