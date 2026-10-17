#!/usr/bin/env bash
${TEIA:-teia} train --export \
  task=graph-node-cls \
  datamodule=graph-node-cls/coco_knn \
  netmodule=graph-node-cls/timm_graphformer \
  netmodule.encoder.pretrained=false \
  epochs=3 \
  data_root=tests/graph-runs/dataset/boards \
  project=graph-tests \
  experiment=graph-node-cls__coco_knn_graphformer \
  runs_root=tests/output/graph/runs \
  "$@"
