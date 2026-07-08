# Relational Grounding Bench

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

This repository tests a targeted question in referring-expression comprehension: does explicit relational context, represented as subject-predicate-object triplets, help a small vision-language model ground expressions that require relational reasoning, compared with attribute-only expressions? The study separates prompt-time relation injection from parameter-efficient fine-tuning to measure whether QLoRA alone closes the relational grounding gap, or whether structured scene context remains useful when the base model is adapted.

## Approach Summary

- Join oracle Visual Genome relation triplets to RefCOCO, RefCOCO+, and RefCOCOg examples through shared COCO image ids.
- Partition expressions into `relational` and `attribute` subsets with a rule-based cue list plus a spaCy dependency fallback.
- Compare four conditions: A zero-shot VLM prompting, B relation-prompted VLM inference, C QLoRA fine-tuning, and D QLoRA fine-tuning with relation context.
- Use stratified `accuracy@IoU-0.5` on relational versus attribute subsets as the headline metric.
- Log fine-tuning memory/time for LoRA rank sweeps and frozen-versus-unfrozen vision tower variants.
- Extend the GraPLUS thesis line from scene graphs for semantic object placement to scene-graph-style relational grounding in a general-purpose VLM, replacing a task-specific GAN+GNN stack with text-facing VLM prompts and QLoRA adaptation.

## Repository Structure

```text
.
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- Makefile
|-- README.md
|-- LICENSE
|-- environment.yaml
|-- configs/
|   `-- config.yaml
|-- common/
|   |-- __init__.py
|   `-- utils.py
|-- data/
|   |-- prepare_refcoco.py
|   |-- join_visual_genome.py
|   |-- classify_expressions.py
|   `-- splits/
|       `-- .gitkeep
|-- prompting/
|   |-- vlm_utils.py
|   |-- zero_shot_baseline.py
|   `-- relation_prompted.py
|-- finetune/
|   |-- train_qlora.py
|   `-- train_qlora_with_context.py
|-- eval/
|   |-- compute_accuracy_iou.py
|   `-- build_results_table.py
|-- notebooks/
|   `-- validate_expression_split.ipynb
|-- tests/
|   |-- __init__.py
|   |-- test_utils.py
|   `-- test_classify_expressions.py
|-- checkpoints/
|   `-- .gitkeep
`-- results/
    `-- .gitkeep
```

## Environment Setup

```bash
conda env create -f environment.yaml && conda activate rgb
python -m spacy download en_core_web_sm
```

The default model is configured in `configs/config.yaml` as `Qwen/Qwen2.5-VL-3B-Instruct` with 4-bit loading enabled and image long-edge resizing capped at 640 pixels.

## Data Preparation

Place local datasets at the paths expected by `configs/config.yaml`. The commands below use `curl`, `unzip`, and POSIX shell syntax from the repository root.

| Source | Required local path | Files expected by the scripts |
| --- | --- | --- |
| RefCOCO / RefCOCO+ / RefCOCOg annotations | `data/raw/refcoco/<dataset>/` | one `refs(*).p` file and `instances.json` per dataset |
| COCO train2014 images | `data/raw/coco/train2014/` | COCO image files referenced by the RefCOCO annotations |
| Visual Genome | `data/raw/visual_genome/` | `relationships.json`, `objects.json`, `image_data.json` |

### Download RefCOCO / RefCOCO+ / RefCOCOg Annotations

The original `lichengunc/refer` API documents the RefCOCO-family downloads, but its README currently notes that the old UNC webserver may be unavailable and points users to the open download-link issue: <https://github.com/lichengunc/refer/issues/14>. Use the official links first; if they fail, download the same three archives from a mirror referenced in that issue and place/extract them into the same target paths.

```bash
mkdir -p data/raw/refcoco

curl -L -o /tmp/refcoco.zip \
  https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
curl -L -o /tmp/refcoco_plus.zip \
  https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
curl -L -o /tmp/refcocog.zip \
  https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

unzip -q /tmp/refcoco.zip -d data/raw/refcoco
unzip -q /tmp/refcoco_plus.zip -d data/raw/refcoco
unzip -q /tmp/refcocog.zip -d data/raw/refcoco
```

After extraction, verify the expected layout:

```bash
find data/raw/refcoco -maxdepth 2 -type f | sort
```

The project expects these files:

```text
data/raw/refcoco/refcoco/instances.json
data/raw/refcoco/refcoco/refs(unc).p
data/raw/refcoco/refcoco+/instances.json
data/raw/refcoco/refcoco+/refs(unc).p
data/raw/refcoco/refcocog/instances.json
data/raw/refcoco/refcocog/refs(umd).p
```

`data/prepare_refcoco.py` requires exactly one `refs(*).p` file per dataset directory. If `refcocog` extracts both Google and UMD split files, keep `refs(umd).p` in `data/raw/refcoco/refcocog/` and move the other `refs(*).p` file out of that directory before running the pipeline.

### Download COCO train2014 Images

```bash
mkdir -p data/raw/coco

curl -L -o data/raw/coco/train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip

unzip -q data/raw/coco/train2014.zip -d data/raw/coco
```

Expected path after extraction:

```text
data/raw/coco/train2014/COCO_train2014_000000000009.jpg
```

### Download Visual Genome Metadata

The relation joiner only needs Visual Genome metadata JSON files, not the Visual Genome image archives.

```bash
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
```

If any Visual Genome extraction command fails, inspect the archive contents manually:

```bash
unzip -l /tmp/vg_image_data.zip
unzip -l /tmp/vg_objects.zip
unzip -l /tmp/vg_relationships.zip
```

Then extract the listed JSON name into the unversioned path expected by the repository.

```bash
unzip -p /tmp/vg_objects.zip <listed-objects-json-name> > data/raw/visual_genome/objects.json
unzip -p /tmp/vg_relationships.zip <listed-relationships-json-name> > data/raw/visual_genome/relationships.json
```

Verify the three required Visual Genome files:

```bash
ls -lh data/raw/visual_genome/image_data.json \
  data/raw/visual_genome/objects.json \
  data/raw/visual_genome/relationships.json
```

### Run the Data Pipeline

Prepare RefCOCO-family JSONL files:

```bash
for dataset in refcoco refcoco+; do
  for split in train val testA testB; do
    python data/prepare_refcoco.py \
      --config configs/config.yaml \
      --dataset "$dataset" \
      --split "$split"
  done
done

for split in train val; do
  python data/prepare_refcoco.py \
    --config configs/config.yaml \
    --dataset refcocog \
    --split "$split"
done
```

Expected outputs:

```text
data/processed/refcoco_<split>.jsonl
data/processed/refcoco+_<split>.jsonl
data/processed/refcocog_<split>.jsonl
```

Join Visual Genome relation triplets by COCO id:

```bash
python data/join_visual_genome.py --config configs/config.yaml
```

Expected output:

```text
data/processed/vg_relations_by_coco_id.jsonl
```

Classify expressions into relational and attribute subsets:

```bash
for dataset in refcoco refcoco+; do
  for split in train val testA testB; do
    python data/classify_expressions.py \
      --config configs/config.yaml \
      --dataset "$dataset" \
      --split "$split"
  done
done

for split in train val; do
  python data/classify_expressions.py \
    --config configs/config.yaml \
    --dataset refcocog \
    --split "$split"
done
```

Expected outputs:

```text
data/splits/<dataset>_<split>_relational.jsonl
data/splits/<dataset>_<split>_attribute.jsonl
data/splits/<dataset>_<split>_classification_log.csv
```

> **Coverage stats:** TODO - fill in after running `join_visual_genome.py`.

## Manual Classifier Validation

Register the environment as a notebook kernel if needed:

```bash
python -m ipykernel install --user --name rgb --display-name "Python (rgb)"
```

Open `notebooks/validate_expression_split.ipynb` in Jupyter or VS Code with the `Python (rgb)` kernel selected. Set `LOG_CSV_PATH` in the first code cell to the split you want to audit, for example:

```python
LOG_CSV_PATH = Path("data/splits/refcoco_val_classification_log.csv")
```

Run the notebook cells, manually fill `human_correct` for the sampled rows, and save the balanced audit annotations to:

```text
results/expression_classifier_audit.csv
```

The notebook samples the classifier log, records `human_correct`, and reports per-label precision for `relational` and `attribute`.

> **Precision (relational / attribute): TODO / TODO** - from `results/expression_classifier_audit.csv`.

## Running the Experiments

The commands below use `refcoco` / `val` as the reporting split. Replace `--dataset`, `--split`, and `--subset` for other datasets or splits. Conditions A and B produce prediction JSONL files directly. Conditions C and D currently train LoRA adapters; scoring C/D requires prediction JSONL files with the schema documented below.

### Full Run Order

After data is downloaded, the intended run order is:

```bash
# 1. Prepare processed RefCOCO-family files and classified splits.
make prepare-data
make classify DATASET=refcoco SPLIT=val

# 2. Perform the manual classifier audit in notebooks/validate_expression_split.ipynb,
#    then paste the measured precision values into this README.

# 3. Run condition A/B inference for both subsets.
make baseline DATASET=refcoco SPLIT=val SUBSET=relational
make baseline DATASET=refcoco SPLIT=val SUBSET=attribute
make prompted DATASET=refcoco SPLIT=val SUBSET=relational
make prompted DATASET=refcoco SPLIT=val SUBSET=attribute

# 4. Train condition C/D adapters.
make finetune DATASET=refcoco LORA_RANK=8
make finetune-context DATASET=refcoco LORA_RANK=8

# 5. Score A/B prediction JSONL files.
make eval CONDITION=A DATASET=refcoco SPLIT=val SUBSET=relational
make eval CONDITION=A DATASET=refcoco SPLIT=val SUBSET=attribute
make eval CONDITION=B DATASET=refcoco SPLIT=val SUBSET=relational \
  PREDICTIONS=results/predictions_condB_refcoco_val_relational_all.jsonl
make eval CONDITION=B DATASET=refcoco SPLIT=val SUBSET=attribute \
  PREDICTIONS=results/predictions_condB_refcoco_val_attribute_all.jsonl

# 6. After C/D prediction JSONLs exist, score them with the explicit
#    eval/compute_accuracy_iou.py commands below.

# 7. Build README-ready tables.
python eval/build_results_table.py --config configs/config.yaml
```

Use the explicit commands in the subsections below when you need exact filenames or ablation settings.

### Condition A: Zero-Shot Baseline

```bash
python prompting/zero_shot_baseline.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split val \
  --subset relational

python prompting/zero_shot_baseline.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split val \
  --subset attribute
```

Default outputs:

```text
results/predictions_condA_refcoco_val_relational.jsonl
results/predictions_condA_refcoco_val_attribute.jsonl
```

### Condition B: Relation-Prompted Inference

```bash
python prompting/relation_prompted.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split val \
  --subset relational

python prompting/relation_prompted.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split val \
  --subset attribute
```

Default outputs:

```text
results/predictions_condB_refcoco_val_relational_all.jsonl
results/predictions_condB_refcoco_val_attribute_all.jsonl
```

Limit injected relations for an ablation:

```bash
python prompting/relation_prompted.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split val \
  --subset relational \
  --max_relations 5
```

### Condition C: QLoRA Fine-Tuning

Train the default rank-8 adapter:

```bash
python finetune/train_qlora.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split train \
  --lora_rank 8 \
  --output_dir checkpoints/qlora_r8
```

LoRA rank sweep:

```bash
for r in 4 8 16; do
  python finetune/train_qlora.py \
    --lora_rank $r \
    --config configs/config.yaml \
    --dataset refcoco \
    --split train \
    --output_dir checkpoints/qlora_r$r
done
```

Frozen-versus-unfrozen vision tower variant:

```bash
python finetune/train_qlora.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split train \
  --lora_rank 8 \
  --freeze_vision_tower \
  --output_dir checkpoints/qlora_r8_frozen

python finetune/train_qlora.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split train \
  --lora_rank 8 \
  --no-freeze_vision_tower \
  --output_dir checkpoints/qlora_r8_unfrozen
```

The training script appends run metadata to `results/finetune_run_log.csv` and saves adapters under `checkpoints/`.

### Condition D: QLoRA Fine-Tuning With Relation Context

```bash
python finetune/train_qlora_with_context.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split train \
  --lora_rank 8 \
  --output_dir checkpoints/qlora_context_r8
```

Condition C/D evaluation is prediction-file based. Adapter inference outputs should follow the same JSONL contract as conditions A/B:

```json
{"ref_id": 123, "expression": "person left of the car", "predicted_bbox": [10, 20, 50, 90], "raw_output": "<box>(10,20),(50,90)</box>"}
```

Use these target names for consistency with the evaluator and result-table builder:

```text
results/predictions_condC_refcoco_val_<subset>.jsonl
results/predictions_condD_refcoco_val_<subset>_all.jsonl
```

There is not yet a separate adapter-inference script in this repository. Until one is added, C/D training is reproducible from the commands above, while C/D evaluation can only be run after generating prediction JSONL files externally or adding an inference wrapper that loads the saved adapter from `checkpoints/`.

## Evaluation

Compute `accuracy@IoU-0.5` for each condition and subset. The evaluator appends rows to `results/accuracy_table.csv`.

```bash
for subset in relational attribute; do
  python eval/compute_accuracy_iou.py \
    --config configs/config.yaml \
    --predictions results/predictions_condA_refcoco_val_${subset}.jsonl \
    --ground_truth data/processed/refcoco_val.jsonl \
    --subset_file data/splits/refcoco_val_${subset}.jsonl \
    --condition A \
    --dataset refcoco \
    --split val \
    --subset $subset
done

for subset in relational attribute; do
  python eval/compute_accuracy_iou.py \
    --config configs/config.yaml \
    --predictions results/predictions_condB_refcoco_val_${subset}_all.jsonl \
    --ground_truth data/processed/refcoco_val.jsonl \
    --subset_file data/splits/refcoco_val_${subset}.jsonl \
    --condition B \
    --dataset refcoco \
    --split val \
    --subset $subset
done

for subset in relational attribute; do
  python eval/compute_accuracy_iou.py \
    --config configs/config.yaml \
    --predictions results/predictions_condC_refcoco_val_${subset}.jsonl \
    --ground_truth data/processed/refcoco_val.jsonl \
    --subset_file data/splits/refcoco_val_${subset}.jsonl \
    --condition C \
    --dataset refcoco \
    --split val \
    --subset $subset
done

for subset in relational attribute; do
  python eval/compute_accuracy_iou.py \
    --config configs/config.yaml \
    --predictions results/predictions_condD_refcoco_val_${subset}_all.jsonl \
    --ground_truth data/processed/refcoco_val.jsonl \
    --subset_file data/splits/refcoco_val_${subset}.jsonl \
    --condition D \
    --dataset refcoco \
    --split val \
    --subset $subset
done
```

Build README-ready tables:

```bash
python eval/build_results_table.py --config configs/config.yaml
```

Expected output:

```text
results/results_table.md
```

## Results

<!-- PASTE results/results_table.md CONTENTS HERE -->

### Key Finding

_TODO: does condition B narrow the accuracy gap vs. A, and how much of the gap does C alone fail to close?_

### Qualitative Examples

<!-- 2-3 examples: relation-prompting fixing a wrong zero-shot prediction; one honest failure case -->

## Ablations

### LoRA Rank Sweep

_TODO: paste table + one-sentence takeaway_

### Number of Injected Relations

_TODO: paste table + one-sentence takeaway_

## Limitations

- RefCOCO/Visual Genome overlap is not complete; exact retained coverage is TODO after running the VG join and expression classifier.
- The rule-based plus spaCy expression classifier has measurable label noise; measured relational and attribute precision are TODO after the manual audit.
- Oracle relation prompting is an upper bound, not a deployable setting, because it assumes ground-truth Visual Genome relations are available at inference time.
- The current repository trains LoRA adapters and evaluates prediction JSONL files; adapter inference for conditions C/D must emit the documented prediction schema before those rows can be scored.

## Relation to Prior Work

This project continues GraPLUS (Khaleghi et al., CVIU 2025), where scene-graph semantics and language-derived embeddings support semantic object placement. Here the same structured-relation hypothesis is moved into referring-expression grounding: relation triplets are exposed as VLM-readable context rather than encoded inside a task-specific GAN+GNN pipeline. The design also reflects the 2025-2026 shift toward using scene graphs as interfaces for VLM reasoning, including [open-world scene-graph generation with VLMs](https://arxiv.org/abs/2506.08189) and [graph-mediated visual grounding](https://arxiv.org/abs/2512.09215).

## Hardware Notes

The intended workstation profile is 8 GB VRAM and 16 GB RAM.

| Component | Hardware implication |
| --- | --- |
| Conditions A/B | Inference-only; lightest experimental conditions, with 4-bit loading and capped image resolution. |
| Conditions C/D | QLoRA fine-tuning with 4-bit NF4, LoRA rank 8-16, frozen vision tower by default, batch size 1, gradient accumulation, and image long-edge cap at 640. |
| Memory logging | `finetune/train_qlora.py` records peak CUDA memory and training time in `results/finetune_run_log.csv`. |

## Citation

If this repository is useful, please cite the GraPLUS thesis work that motivates the structured-relation framing:

```bibtex
@article{khaleghi2025graplus,
  title={GraPLUS: Graph-based Placement Using Semantics for image composition},
  author={Khaleghi, Mir Mohammad and Safayani, Mehran and Mirzaei, Abdolreza},
  journal={Computer Vision and Image Understanding},
  volume={259},
  pages={104427},
  year={2025},
  month={September},
  publisher={Elsevier},
  doi={10.1016/j.cviu.2025.104427},
  url={https://www.sciencedirect.com/science/article/abs/pii/S107731422500150X}
}
```

Dataset citation notes:

- RefCOCO, RefCOCO+, and RefCOCOg: referring-expression datasets distributed through the UNC `refer` interface and related dataset releases: <https://github.com/lichengunc/refer>.
- Visual Genome: source of object and relationship annotations used for oracle relation triplets: <https://visualgenome.org/>.
- COCO train2014: source images for the RefCOCO-family annotations: <https://cocodataset.org/#download>.

## License

MIT. See [LICENSE](LICENSE).

## Contact

Mir Mohammad Khaleghi (email placeholder)
