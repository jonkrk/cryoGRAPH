#!/bin/bash
# To run a batch of configs, add path to batch directory.
# For instance: ./scripts/run_cfg_batch.sh ./cfgs/FOLDER/*
CFGS="$@"
for f in $CFGS
do
  echo "Training with config: $f"
  uv run main.py -c "$f"
done