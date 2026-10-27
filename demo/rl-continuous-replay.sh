#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

teia train \
  task=rl-continuous-replay \
  project=demo \
  experiment=rl-continuous-replay \
  datamodule.runner.total_env_steps=${STEPS:-12000} \
  datamodule.roll.warmup_steps=1000 \
  "$@"
