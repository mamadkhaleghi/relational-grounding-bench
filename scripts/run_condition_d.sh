#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Condition D: QLoRA Fine-Tuning With Relation Context".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python finetune/train_qlora_with_context.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --split train \
  --max_train_samples 4000 \
  --lora_rank 8 \
  --output_dir checkpoints/qlora_context_r8

for subset in relational positional attribute; do
  env -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u http_proxy \
      -u HTTPS_PROXY -u https_proxy \
      python prompting/finetuned_inference.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --subset "$subset" \
    --adapter_dir checkpoints/qlora_context_r8 \
    --condition D
done
