#!/usr/bin/env bash
set -euo pipefail
# Replaces README section: "Download RefCOCO / RefCOCO+ / RefCOCOg Annotations".

CONFIG="${CONFIG:-configs/config.yaml}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-val}"

mkdir -p data/raw/refcoco

curl -L -o /tmp/refcoco.zip \
  "https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"

curl -L -o /tmp/refcoco_plus.zip \
  "https://web.archive.org/web/20220413011656id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"

curl -L -o /tmp/refcocog.zip \
  "https://web.archive.org/web/20220413012904id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"

unzip -q /tmp/refcoco.zip -d data/raw/refcoco
unzip -q /tmp/refcoco_plus.zip -d data/raw/refcoco
unzip -q /tmp/refcocog.zip -d data/raw/refcoco
