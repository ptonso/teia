#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/audio_cls.py data/audio-multi-cls --multi-label

teia train \
  task=audio-multi-cls \
  project=demo \
  experiment=audio-multi-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/audio-multi-cls \
  "$@"
