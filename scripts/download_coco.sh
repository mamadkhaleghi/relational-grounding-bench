#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Download COCO train2014 Images".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

mkdir -p data/raw/coco

curl -L -o data/raw/coco/train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip

unzip -q data/raw/coco/train2014.zip -d data/raw/coco
