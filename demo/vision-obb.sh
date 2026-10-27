#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/shapes.py data/vision-obb --format obb

teia train \
  task=vision-obb \
  project=demo \
  experiment=vision-obb \
  epochs=${EPOCHS:-30} \
  data_root=data/vision-obb \
  "$@"
