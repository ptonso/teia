#!/usr/bin/env bash
${TEIA:-teia} train --export \
  task=graph-node-cls \
  datamodule=graph-node-cls/labelme_graph \
  netmodule=graph-node-cls/timm_no_graph \
  netmodule.encoder.pretrained=false \
  epochs=3 \
  data_root=tests/graph-runs/dataset/boards \
  project=graph-tests \
  experiment=graph-node-cls__labelme_no_graph \
  runs_root=tests/output/graph/runs \
  "$@"
