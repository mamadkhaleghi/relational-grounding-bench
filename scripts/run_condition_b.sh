#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Condition B: Relation-Prompted Inference".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

for subset in relational positional attribute; do
  env -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u http_proxy \
      -u HTTPS_PROXY -u https_proxy \
      python prompting/relation_prompted.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --subset "$subset"
done
