#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare_digits.py data/digits

teia train \
  datamodule=vision-cls/image_folder \
  netmodule=vision-cls/mlp \
  evalmodule=vision-cls/default \
  project=demo \
  experiment=digits \
  epochs=20 \
  data_root=data/digits \
  "$@"
