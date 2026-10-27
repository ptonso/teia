#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/vision_sem_seg.py data/vision-sem-seg

teia train \
  task=vision-sem-seg \
  project=demo \
  experiment=vision-sem-seg \
  epochs=${EPOCHS:-20} \
  data_root=data/vision-sem-seg \
  "$@"
