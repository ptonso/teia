#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare/shapes.py data/vision-det --format det

teia train \
  task=vision-det \
  project=demo \
  experiment=vision-det \
  epochs=${EPOCHS:-30} \
  data_root=data/vision-det \
  "$@"
