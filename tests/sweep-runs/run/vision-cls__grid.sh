#!/usr/bin/env bash
${TEIA:-teia} sweep \
  task=vision-cls \
  netmodule.input.image_size=32 \
  datamodule.batch_size=2 \
  epochs=3 \
  data_root=tests/vision-runs/dataset/class-dir/oxford-iiit-pet \
  project=sweep-tests \
  experiment=vision-cls__grid \
  runs_root=tests/output/sweep/runs \
  sweep.objective=val/loss sweep.mode=min \
  'netmodule.lr=1e-4,1e-3' \
  "$@"
