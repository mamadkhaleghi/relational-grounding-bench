#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Full Run Order".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

# 1. Prepare processed RefCOCO-family files and classified splits.
make prepare-data CONFIG="$CONFIG"
make classify CONFIG="$CONFIG" DATASET="$DATASET" SPLIT="$SPLIT"

# 2. Perform the manual classifier audit in notebooks/validate_expression_split.ipynb,
#    then paste the measured precision values into this README.
if [[ "${SKIP_MANUAL_AUDIT:-0}" != "1" ]]; then
  if [[ ! -t 0 ]]; then
    echo "Manual classifier audit required before experiments continue." >&2
    echo "Run this script interactively, or set SKIP_MANUAL_AUDIT=1 only if the audit is already complete." >&2
    exit 1
  fi
  echo "Complete the classifier audit in notebooks/validate_expression_split.ipynb."
  read -r -p "After saving the audit results and updating README.md, press Enter to continue: "
fi

# 3. Run condition A/B inference for all three subsets.
CONFIG="$CONFIG" DATASET="$DATASET" SPLIT="$SPLIT" scripts/run_condition_a.sh
CONFIG="$CONFIG" DATASET="$DATASET" SPLIT="$SPLIT" scripts/run_condition_b.sh

# 4. Train condition C/D adapters.
# 5. Run condition C/D adapter inference for all three subsets.
CONFIG="$CONFIG" DATASET="$DATASET" SPLIT="$SPLIT" scripts/run_condition_c.sh
CONFIG="$CONFIG" DATASET="$DATASET" SPLIT="$SPLIT" scripts/run_condition_d.sh

# 6. Score A/B prediction JSONL files.
# 7. After C/D prediction JSONLs exist, score them with the explicit
#    eval/compute_accuracy_iou.py commands below.
# 8. Build README-ready tables.
CONFIG="$CONFIG" DATASET="$DATASET" SPLIT="$SPLIT" scripts/evaluate_all.sh
