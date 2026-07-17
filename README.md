# Relational Grounding Bench

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

This repository tests a targeted question in referring-expression comprehension: does explicit relational context, represented as subject-predicate-object triplets, help a small vision-language model ground expressions that require relational reasoning, compared with expressions that rely on frame position or attributes only? The study separates prompt-time relation injection from parameter-efficient fine-tuning to measure whether QLoRA alone closes the relational grounding gap, or whether structured scene context remains useful when the base model is adapted.

## Approach Summary

- Join oracle Visual Genome relation triplets to RefCOCO, RefCOCO+, and RefCOCOg examples through shared COCO image ids.
- Partition expressions into three categories with a rule-based cue list plus a spaCy dependency fallback: `relational` expressions require reasoning about a second, distinct, Visual-Genome-annotated object; `positional` expressions require frame-position or ordinal reasoning, such as "right bear", but no second object, and Visual Genome relation triplets cannot help ground these because VG does not annotate frame position; `attribute` expressions require no spatial reasoning at all, such as "red boat".
- This three-way split was added after manual audit showed frame-position language was a large, systematic category that did not fit cleanly into either original bucket.
- Compare four conditions: A zero-shot VLM prompting, B relation-prompted VLM inference, C QLoRA fine-tuning, and D QLoRA fine-tuning with relation context.
- Use stratified `accuracy@IoU-0.5` on relational, positional, and attribute subsets as the headline metric.
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
|   |-- relation_prompted.py
|   `-- finetuned_inference.py
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
|-- scripts/
|   |-- ablation_lora_rank_sweep.sh
|   |-- ablation_vision_tower.sh
|   |-- download_coco.sh
|   |-- download_refcoco.sh
|   |-- download_visual_genome.sh
|   |-- evaluate_all.sh
|   |-- prepare_data.sh
|   |-- run_condition_a.sh
|   |-- run_condition_b.sh
|   |-- run_condition_c.sh
|   |-- run_condition_d.sh
|   `-- run_full_pipeline.sh
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

The original UNC webserver (`bvisionweb1.cs.unc.edu`) has been down since 2022 ([tracked here](https://github.com/lichengunc/refer/issues/14)). The commands below use a Wayback Machine snapshot of the same three archives instead.

```bash
bash scripts/download_refcoco.sh
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
data/raw/refcoco/refcocog/refs(google).p
data/raw/refcoco/refcocog/refs(umd).p
```

Both `refcocog` files are kept in place; `data/prepare_refcoco.py`'s `--split_by {umd,google}` flag (default: `umd`) selects between them explicitly, so no file moving is needed.

### Download COCO train2014 Images

Download the COCO train2014 archive with:

```bash
bash scripts/download_coco.sh
```

Expected path after extraction:

```text
data/raw/coco/train2014/COCO_train2014_000000000009.jpg
```

### Download Visual Genome Metadata

The relation joiner only needs Visual Genome metadata JSON files, not the Visual Genome image archives.

```bash
bash scripts/download_visual_genome.sh
```

Verify the three required Visual Genome files:

```bash
ls -lh data/raw/visual_genome/image_data.json \
  data/raw/visual_genome/objects.json \
  data/raw/visual_genome/relationships.json
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

### Run the Data Pipeline

Prepare the RefCOCO-family JSONL files, join Visual Genome relation triplets, and classify expressions with:

```bash
bash scripts/prepare_data.sh
```

RefCOCOg ships two split protocols in the same annotation folder (`refs(umd).p` and
`refs(google).p`); `--split_by umd` selects the standard protocol with full train/val/test
coverage used across the referring-expression literature.

Explicit RefCOCOg UMD test command:

```bash
python data/prepare_refcoco.py --config configs/config.yaml --dataset refcocog --split test --split_by umd
```

Expected outputs:

```text
data/processed/refcoco_<split>.jsonl
data/processed/refcoco+_<split>.jsonl
data/processed/refcocog_<split>.jsonl
data/processed/refcocog_test.jsonl
```

Verified row counts (expression rows written, 0 failed resolves):

| Dataset | train | val | test | testA | testB |
| --- | ---: | ---: | ---: | ---: | ---: |
| refcoco | 120624 | 10834 | - | 5657 | 5095 |
| refcoco+ | 120191 | 10758 | - | 5726 | 4889 |
| refcocog (umd) | 80512 | 4896 | 9602 | - | - |

Join Visual Genome relation triplets by COCO id (included in the script above):

Expected output:

```text
data/processed/vg_relations_by_coco_id.jsonl
```

Classify expressions into relational, positional, and attribute subsets (also included in the script above):

Expected outputs:

```text
data/splits/<dataset>_<split>_relational.jsonl
data/splits/<dataset>_<split>_positional.jsonl
data/splits/<dataset>_<split>_attribute.jsonl
data/splits/<dataset>_<split>_classification_log.csv
data/splits/refcocog_test_relational.jsonl
data/splits/refcocog_test_positional.jsonl
data/splits/refcocog_test_attribute.jsonl
data/splits/refcocog_test_classification_log.csv
```

> **Coverage stats:** VG-relation coverage by dataset/split.

| Dataset | Split | Coverage | Matched / Total |
|---|---:|---:|---:|
| refcoco | train | 38.38% | 46294 / 120624 |
| refcoco | val | 37.37% | 4049 / 10834 |
| refcoco | testA | 36.31% | 2054 / 5657 |
| refcoco | testB | 37.11% | 1891 / 5095 |
| refcoco+ | train | 38.28% | 46010 / 120191 |
| refcoco+ | val | 37.55% | 4040 / 10758 |
| refcoco+ | testA | 36.81% | 2108 / 5726 |
| refcoco+ | testB | 36.45% | 1782 / 4889 |
| refcocog (umd) | train | 38.20% | 30758 / 80512 |
| refcocog (umd) | val | 38.30% | 1875 / 4896 |
| refcocog (umd) | test | 39.23% | 3767 / 9602 |

### Classifier Validation History

The expression classifier was refined through four audit rounds with measured precision for each active label. Round 2 showed that reducing false positives from the original broad spaCy fallback exposed a hidden false-negative problem: frame-position language such as "right bear" and "left man" did not fit either original category. That evidence motivated the 3-way split in round 3, and round 4 closed the remaining audit-identified gaps around phrasal verbs, camera/frame references, and missing position-word forms.

| Round | Change | attribute | positional | relational |
|---|---|---:|---:|---:|
| 1 | Original keyword classifier (2-way: relational/attribute) | 0.390 | n/a | 0.860 |
| 2 | Stricter spaCy fallback + with/attire disambiguation (still 2-way) | 0.350 | n/a | 0.960 |
| 3 | Introduced 3-way split (relational/positional/attribute) | ~0.84 | ~0.99 | ~0.91-0.93 |
| 4 | Fixed phrasal-verb "over", viewer/frame references, leftmost/rightmost/front/back gaps | ~0.98-0.99 | ~0.99-1.00 | ~0.98-0.99 |

This audit trail is part of the experimental methodology: category definitions were revised only when measured errors indicated a systematic semantic distinction that mattered for the relational-grounding comparison.

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

The notebook samples the classifier log, records `human_correct`, and reports per-label precision for `relational`, `positional`, and `attribute`.

> **Precision (relational / positional / attribute):** see the Classifier Validation History
table above. Rounds 1-2 were measured via the notebook's interactive `human_correct` review loop
saved to `results/expression_classifier_audit.csv`; rounds 3-4 were estimated via structured
manual review of exported stratified samples (100 per label, via the notebook's
export-for-review cells) rather than that saved-CSV mechanism, which is why those figures are
reported as approximate ranges rather than exact values.

## Running the Experiments

The commands below use `refcoco` / `val` as the reporting split. Replace `--dataset`, `--split`, and `--subset` for other datasets or splits. Conditions A and B produce prediction JSONL files directly. Conditions C and D currently train LoRA adapters; scoring C/D requires prediction JSONL files with the schema documented below. A local proxy environment variable can hang or time out Hugging Face downloads/loads, so model-loading commands unset those variables for their duration.

To fit available compute on an 8 GB GPU, training uses a fixed, seeded 4,000-example subset of the roughly 120,000-example train split rather than the full split, for which the measured training rate projected approximately 37.4 hours per epoch.

### Full Run Order

After data is downloaded, the intended run order is:

1. Prepare processed RefCOCO-family files and classified splits.
2. Perform the manual classifier audit in notebooks/validate_expression_split.ipynb, then paste the measured precision values into this README.
3. Run condition A/B inference for all three subsets.
4. Train condition C/D adapters.
5. Run condition C/D adapter inference for all three subsets.
6. Score A/B prediction JSONL files.
7. After C/D prediction JSONLs exist, score them with the evaluation script below.
8. Build README-ready tables.

Run the full sequence from the repository root with:

```bash
bash scripts/run_full_pipeline.sh
```

The full-pipeline script pauses after data preparation so the manual classifier audit can be completed before experiments continue. If the audit has already been completed and its results are saved, bypass only that pause with `SKIP_MANUAL_AUDIT=1 bash scripts/run_full_pipeline.sh`.

Use the script references in the subsections below when you need the standard runs or ablation settings.

### Condition A: Zero-Shot Baseline

Run the zero-shot baseline for all three subsets with:

```bash
bash scripts/run_condition_a.sh
```

Default outputs:

```text
results/predictions_condA_refcoco_val_relational.jsonl
results/predictions_condA_refcoco_val_positional.jsonl
results/predictions_condA_refcoco_val_attribute.jsonl
```

### Condition B: Relation-Prompted Inference

Run relation-prompted inference for all three subsets with:

```bash
bash scripts/run_condition_b.sh
```

Default outputs:

```text
results/predictions_condB_refcoco_val_relational_all.jsonl
results/predictions_condB_refcoco_val_positional_all.jsonl
results/predictions_condB_refcoco_val_attribute_all.jsonl
```

Limit injected relations for an ablation:

Use this one-off command because `scripts/run_condition_b.sh` runs the default uncapped relation count: `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy python prompting/relation_prompted.py --config configs/config.yaml --dataset refcoco --split val --subset relational --max_relations 5`.

### Smoke Tests and Proxy Troubleshooting

Conditions A and B accept `--limit_samples N` for deterministic smoke tests on the first `N`
examples in the selected subset. Smoke-test outputs append `_smoketest` before `.jsonl`, so they
cannot silently overwrite full-run predictions. For example:

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python prompting/zero_shot_baseline.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split val \
  --subset relational \
  --limit_samples 10
```

If Hugging Face model loading fails with `Unknown scheme for proxy URL URL('socks://...')`, the
current shell has proxy environment variables that HTTPX cannot use. If a proxy is not required,
disable those variables for only the smoke-test invocation:

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python prompting/zero_shot_baseline.py \
      --config configs/config.yaml \
      --dataset refcoco \
      --split val \
      --subset relational \
      --limit_samples 10
```

This does not change the parent shell environment. If a SOCKS proxy is required, configure it
with a supported `socks5://` URL and install HTTPX SOCKS support in the active environment.

### Condition C: QLoRA Fine-Tuning

Train the default rank-8 adapter and run inference for all three subsets with:

```bash
bash scripts/run_condition_c.sh
```

Run the LoRA rank sweep with:

```bash
bash scripts/ablation_lora_rank_sweep.sh
```

Run the frozen-versus-unfrozen vision tower variant with:

```bash
bash scripts/ablation_vision_tower.sh
```

The base-run script above also runs inference with the default rank-8 adapter for all three subsets.

Default outputs:

```text
results/predictions_condC_refcoco_val_relational.jsonl
results/predictions_condC_refcoco_val_positional.jsonl
results/predictions_condC_refcoco_val_attribute.jsonl
```

Both QLoRA training scripts save resumable Hugging Face Trainer checkpoints every 50 update steps by default. Use `--save_steps N` to change the interval and `--save_total_limit N` to cap retained checkpoints (default: 3). Checkpoints include optimizer and scheduler state; the final LoRA adapter is still saved directly in `--output_dir` after training completes.

`scripts/run_condition_c.sh` accepts only the `CONFIG`, `DATASET`, and `SPLIT` environment overrides; it does not forward a resume flag, and there is no separate resume-specific script. To resume that training run, invoke the underlying trainer directly with the same output directory and `--resume_from_checkpoint auto`:

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python finetune/train_qlora.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split train \
  --max_train_samples 4000 \
  --lora_rank 8 \
  --output_dir checkpoints/qlora_r8 \
  --resume_from_checkpoint auto
```

After the resumed training finishes, run the inference loop from `scripts/run_condition_c.sh` separately; rerunning the wrapper itself would start training afresh.

```bash
for subset in relational positional attribute; do
  env -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u http_proxy \
      -u HTTPS_PROXY -u https_proxy \
      python prompting/finetuned_inference.py \
    --config configs/config.yaml \
    --dataset refcoco \
    --split val \
    --subset "$subset" \
    --adapter_dir checkpoints/qlora_r8 \
    --condition C
done
```

For the same run through the Makefile, use `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy make finetune DATASET=refcoco LORA_RANK=8 RESUME_FROM_CHECKPOINT=auto`.

`auto` selects the latest `checkpoint-<step>` directory under `--output_dir`, or starts fresh if none exists. To select one checkpoint explicitly, pass its path instead, for example `--resume_from_checkpoint checkpoints/qlora_r8/checkpoint-150`. Omitting `--resume_from_checkpoint` starts a fresh run. The training scripts log the selected mode, resolved checkpoint path, and resume step when available. Completed resumed runs also record the resolved checkpoint in the `resumed_from` column of `results/finetune_run_log.csv`; existing rows from the older five-column format are preserved with an empty value in that column.

The training scripts append completed-run metadata to `results/finetune_run_log.csv` and save adapters under `checkpoints/`.

### Condition D: QLoRA Fine-Tuning With Relation Context

Train the default rank-8 context adapter and run inference for all three subsets with:

```bash
bash scripts/run_condition_d.sh
```

`scripts/run_condition_d.sh` accepts only the `CONFIG`, `DATASET`, and `SPLIT` environment overrides; it does not forward a resume flag, and there is no separate resume-specific script. To resume that training run, invoke the underlying context trainer directly with the same output directory and `--resume_from_checkpoint auto`:

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python finetune/train_qlora_with_context.py \
  --config configs/config.yaml \
  --dataset refcoco \
  --split train \
  --max_train_samples 4000 \
  --lora_rank 8 \
  --max_relations 10 \
  --output_dir checkpoints/qlora_context_r8 \
  --resume_from_checkpoint auto
```

After the resumed training finishes, run the inference loop from `scripts/run_condition_d.sh` separately; rerunning the wrapper itself would start training afresh.

```bash
for subset in relational positional attribute; do
  env -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u http_proxy \
      -u HTTPS_PROXY -u https_proxy \
      python prompting/finetuned_inference.py \
    --config configs/config.yaml \
    --dataset refcoco \
    --split val \
    --subset "$subset" \
    --adapter_dir checkpoints/qlora_context_r8 \
    --condition D
done
```

For the same context run through the Makefile, use `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy make finetune-context DATASET=refcoco LORA_RANK=8 RESUME_FROM_CHECKPOINT=auto`.

`auto` selects the latest `checkpoint-<step>` directory under `--output_dir`, or starts fresh if none exists. To select one checkpoint explicitly, pass its path instead, for example `--resume_from_checkpoint checkpoints/qlora_context_r8/checkpoint-150`. Omitting `--resume_from_checkpoint` starts a fresh run. The training scripts log the selected mode, resolved checkpoint path, and resume step when available. Completed resumed runs also record the resolved checkpoint in the `resumed_from` column of `results/finetune_run_log.csv`; existing rows from the older five-column format are preserved with an empty value in that column.

The base-run script above also runs inference with the default rank-8 context adapter for all three subsets.

Default outputs:

```text
results/predictions_condD_refcoco_val_relational_all.jsonl
results/predictions_condD_refcoco_val_positional_all.jsonl
results/predictions_condD_refcoco_val_attribute_all.jsonl
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

`--max_relations 10` caps the number of injected Visual Genome relation triplets per training example. This was added after a CUDA OOM at step 318/1500, caused by one training image having an unusually large annotated relation count; it is a training-time memory safeguard not present in condition B's inference.

## Evaluation

Compute `accuracy@IoU-0.5` for each condition and subset. The evaluator appends rows to `results/accuracy_table.csv`.

Run the complete evaluation and results-table build with:

```bash
bash scripts/evaluate_all.sh
```

The evaluation script also builds README-ready tables.

Expected output:

```text
results/results_table.md
```

## Results

## Accuracy by Condition

| condition | relational | positional | attribute | gap | gap_positional |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.6886 | 0.8771 | 0.8616 | 0.1730 | -0.0155 |
| B | 0.6557 | 0.8526 | 0.8577 | 0.2020 | 0.0051 |
| C | 0.7899 | 0.9117 | 0.9355 | 0.1456 | 0.0238 |
| D | 0.7848 | 0.9079 | 0.9292 | 0.1444 | 0.0213 |

## LoRA Rank-Sweep Ablation

| rank | accuracy | peak_mem_mb | train_seconds |
| ---: | ---: | ---: | ---: |
| 8 | NA | 5398.96 | 11136.06 |

Ranks 4 and 16, and the frozen/unfrozen vision-tower variant, are pending -- see the Optional Ablations section of the training plan.

### Key Finding

Explicit Visual-Genome relation-triplet context did not narrow the relational-vs-attribute
accuracy gap, in either the zero-shot or fine-tuned setting. Relation-prompting (B) actually
widened the gap relative to the zero-shot baseline (A) (0.1730 -> 0.2020), driven by a drop in
relational accuracy (0.6886 -> 0.6557) rather than any attribute-side change. QLoRA fine-tuning
alone (C, no relation context) closed roughly 16% of the zero-shot gap (0.1730 -> 0.1456) and
produced the largest absolute relational-accuracy gain of any condition (+0.1013 over A). Adding
relation context during fine-tuning (D) did not improve on C: every subset's accuracy was
marginally lower under D than C (relational -0.0051, positional -0.0038, attribute -0.0063), and
D's gap (0.1444) was essentially unchanged from C's (0.1456). In this setup, structured
relational context gave no measurable benefit over fine-tuning alone at either inference time
(B vs. A) or training time (D vs. C) -- and in the zero-shot case, it appears to modestly hurt
relational grounding rather than help it. Sample sizes (395 relational / 2368 positional / 1286
attribute) mean these differences, while directionally consistent across both context
conditions, should not be read as large-effect-size results without a proper significance test.

### Qualitative Examples

<!-- 2-3 examples: relation-prompting fixing a wrong zero-shot prediction; one honest failure case -->

## Ablations

### LoRA Rank Sweep

Not run for this submission -- deliberately deferred; exact commands are documented in the
Optional Ablations section of the training session plan and can be added in a future update.

### Number of Injected Relations

Not run for this submission -- deliberately deferred; exact commands are documented in the
Optional Ablations section of the training session plan and can be added in a future update.

## Limitations

Condition D's training caps injected relation context at 10 triplets per example (`--max_relations 10`, added for memory safety), while condition B's inference used the default, uncapped relation count. This is a real asymmetry between the two relation-context conditions: any accuracy difference observed between B and D is not purely attributable to prompting vs. fine-tuning, since D was also given less relation context per example than B. This could be closed by rerunning condition B with `--max_relations 10` for a strictly matched comparison.

- RefCOCO/Visual Genome overlap is not complete -- retained coverage ranges from 36.31% to 39.23% depending on dataset/split, see the Coverage stats table above.
- The rule-based plus spaCy expression classifier has measurable label noise; audit precision is documented above and should be rechecked whenever classifier rules or dataset coverage change.
- Visual Genome relation triplets cannot ground frame-position-only expressions, so those were split into their own `positional` category rather than left mixed into `attribute` or `relational`, a design decision empirically motivated by successive manual audits.
- Oracle relation prompting is an upper bound, not a deployable setting, because it assumes ground-truth Visual Genome relations are available at inference time.

## Relation to Prior Work

This project continues GraPLUS (Khaleghi et al., CVIU 2025), where scene-graph semantics and language-derived embeddings support semantic object placement. Here the same structured-relation hypothesis is moved into referring-expression grounding: relation triplets are exposed as VLM-readable context rather than encoded inside a task-specific GAN+GNN pipeline. The design also reflects the 2025-2026 shift toward using scene graphs as interfaces for VLM reasoning, including [open-world scene-graph generation with VLMs](https://arxiv.org/abs/2506.08189) and [graph-mediated visual grounding](https://arxiv.org/abs/2512.09215).

## Hardware Notes

The intended workstation profile is 8 GB VRAM and 16 GB RAM.

Peak VRAM for QLoRA fine-tuning was empirically confirmed at 5,398.96 MB (of 8,188 MB
available) for rank 8 on this exact 8GB card, logged automatically to
`results/finetune_run_log.csv`.

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
