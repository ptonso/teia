#!/usr/bin/env bash
${TEIA:-teia} train \
  task=vision-cls \
  netmodule.input.image_size=32 \
  datamodule.batch_size=2 \
  epochs=3 \
  data_root=tests/vision-runs/dataset/class-dir/oxford-iiit-pet \
  project=vision-tests \
  experiment=vision-cls__n_runs \
  runs_root=tests/output/vision/runs \
  runtime.n_runs=2 \
  "$@"
