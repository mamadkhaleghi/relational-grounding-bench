#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Condition C: QLoRA Fine-Tuning" (vision-tower ablation).

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python finetune/train_qlora.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --split train \
  --max_train_samples 4000 \
  --lora_rank 8 \
  --freeze_vision_tower \
  --output_dir checkpoints/qlora_r8_frozen

env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python finetune/train_qlora.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --split train \
  --max_train_samples 4000 \
  --lora_rank 8 \
  --no-freeze_vision_tower \
  --output_dir checkpoints/qlora_r8_unfrozen
