#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Evaluation".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

for subset in relational positional attribute; do
  python eval/compute_accuracy_iou.py \
    --config "$CONFIG" \
    --predictions "results/predictions_condA_${DATASET}_${SPLIT}_${subset}.jsonl" \
    --ground_truth "data/processed/${DATASET}_${SPLIT}.jsonl" \
    --subset_file "data/splits/${DATASET}_${SPLIT}_${subset}.jsonl" \
    --condition A \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --subset "$subset"
done

for subset in relational positional attribute; do
  python eval/compute_accuracy_iou.py \
    --config "$CONFIG" \
    --predictions "results/predictions_condB_${DATASET}_${SPLIT}_${subset}_all.jsonl" \
    --ground_truth "data/processed/${DATASET}_${SPLIT}.jsonl" \
    --subset_file "data/splits/${DATASET}_${SPLIT}_${subset}.jsonl" \
    --condition B \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --subset "$subset"
done

for subset in relational positional attribute; do
  python eval/compute_accuracy_iou.py \
    --config "$CONFIG" \
    --predictions "results/predictions_condC_${DATASET}_${SPLIT}_${subset}.jsonl" \
    --ground_truth "data/processed/${DATASET}_${SPLIT}.jsonl" \
    --subset_file "data/splits/${DATASET}_${SPLIT}_${subset}.jsonl" \
    --condition C \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --subset "$subset"
done

for subset in relational positional attribute; do
  python eval/compute_accuracy_iou.py \
    --config "$CONFIG" \
    --predictions "results/predictions_condD_${DATASET}_${SPLIT}_${subset}_all.jsonl" \
    --ground_truth "data/processed/${DATASET}_${SPLIT}.jsonl" \
    --subset_file "data/splits/${DATASET}_${SPLIT}_${subset}.jsonl" \
    --condition D \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --subset "$subset"
done

python eval/build_results_table.py --config "$CONFIG"
