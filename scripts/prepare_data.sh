#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Run the Data Pipeline".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

for dataset in refcoco refcoco+; do
  for split in train val testA testB; do
    python data/prepare_refcoco.py \
      --config "$CONFIG" \
      --dataset "$dataset" \
      --split "$split"
  done
done

for split in train val test; do
  python data/prepare_refcoco.py \
    --config "$CONFIG" \
    --dataset refcocog \
    --split "$split" \
    --split_by umd
done

python data/join_visual_genome.py --config "$CONFIG"

for dataset in refcoco refcoco+; do
  for split in train val testA testB; do
    python data/classify_expressions.py \
      --config "$CONFIG" \
      --dataset "$dataset" \
      --split "$split"
  done
done

for split in train val test; do
  python data/classify_expressions.py \
    --config "$CONFIG" \
    --dataset refcocog \
    --split "$split"
done
