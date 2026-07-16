#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Condition C: QLoRA Fine-Tuning" (LoRA rank sweep).

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

for r in 4 8 16; do
  env -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u http_proxy \
      -u HTTPS_PROXY -u https_proxy \
      python finetune/train_qlora.py \
    --lora_rank $r \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --split train \
    --max_train_samples 4000 \
    --output_dir checkpoints/qlora_r$r
done
