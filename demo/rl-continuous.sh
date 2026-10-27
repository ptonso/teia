#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

teia train \
  task=rl-continuous \
  project=demo \
  experiment=rl-continuous \
  datamodule.runner.total_env_steps=${STEPS:-12000} \
  "$@"
