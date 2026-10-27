#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/vision_cls.py data/vision-gen

teia train \
  task=vision-gen \
  project=demo \
  experiment=vision-gen \
  epochs=${EPOCHS:-50} \
  data_root=data/vision-gen \
  "$@"
