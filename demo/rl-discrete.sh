#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

teia train \
  task=rl-discrete \
  project=demo \
  experiment=rl-discrete \
  datamodule.runner.total_env_steps=${STEPS:-20000} \
  datamodule.roll.warmup_steps=1000 \
  "$@"
