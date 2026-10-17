#!/usr/bin/env bash
${TEIA:-teia} train --export \
  task=graph-node-cls \
  datamodule=graph-node-cls/coco_graph \
  netmodule=graph-node-cls/timm_gatv2 \
  netmodule.encoder.pretrained=false \
  epochs=3 \
  data_root=tests/graph-runs/dataset/boards \
  project=graph-tests \
  experiment=graph-node-cls__coco_gatv2 \
  runs_root=tests/output/graph/runs \
  "$@"
