#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare_tabular.py data/tabular

teia train \
  datamodule=tabular-cls/csv \
  netmodule=tabular-cls/numerical_mlp \
  evalmodule=tabular-cls/default \
  project=demo \
  experiment=tabular \
  epochs=30 \
  data_root=data/tabular \
  "$@"
