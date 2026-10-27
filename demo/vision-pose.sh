#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/shapes.py data/vision-pose --format pose

teia train \
  task=vision-pose \
  project=demo \
  experiment=vision-pose \
  epochs=${EPOCHS:-30} \
  data_root=data/vision-pose \
  "$@"
