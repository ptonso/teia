#!/usr/bin/env bash
${TEIA:-teia} sweep \
  task=vision-cls \
  netmodule.input.image_size=32 \
  datamodule.batch_size=2 \
  epochs=3 \
  data_root=tests/vision-runs/dataset/class-dir/oxford-iiit-pet \
  project=sweep-tests \
  experiment=vision-cls__random_n_runs \
  runs_root=tests/output/sweep/runs \
  sweep.objective=val/loss sweep.mode=min sweep.n_trials=3 \
  'netmodule.lr=tag(log,interval(1e-4,1e-2))' runtime.n_runs=2 \
  "$@"
