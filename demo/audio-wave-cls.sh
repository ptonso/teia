#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/audio_cls.py data/audio

teia train \
  task=audio-wave-cls \
  project=demo \
  experiment=audio-wave-cls \
  epochs=${EPOCHS:-20} \
  data_root=data/audio \
  "$@"
