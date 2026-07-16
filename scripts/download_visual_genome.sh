#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Download Visual Genome Metadata".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

mkdir -p data/raw/visual_genome

curl -L -o /tmp/vg_image_data.zip \
  https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
curl -L -o /tmp/vg_objects.zip \
  https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip
curl -L -o /tmp/vg_relationships.zip \
  https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1_2.json.zip

unzip -p /tmp/vg_image_data.zip \
  "$(unzip -Z1 /tmp/vg_image_data.zip | grep 'image_data.*json$' | head -n 1)" \
  > data/raw/visual_genome/image_data.json
unzip -p /tmp/vg_objects.zip \
  "$(unzip -Z1 /tmp/vg_objects.zip | grep 'objects.*json$' | head -n 1)" \
  > data/raw/visual_genome/objects.json
unzip -p /tmp/vg_relationships.zip \
  "$(unzip -Z1 /tmp/vg_relationships.zip | grep 'relationships.*json$' | head -n 1)" \
  > data/raw/visual_genome/relationships.json
