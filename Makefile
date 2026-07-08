.PHONY: env prepare-data classify baseline prompted finetune finetune-context eval results test lint

PYTHON ?= python
CONFIG ?= configs/config.yaml

# Override dataset/split/subset on the command line, e.g. `make baseline DATASET=refcocog SPLIT=val SUBSET=attribute`.
DATASET ?= refcoco
SPLIT ?= val
SUBSET ?= relational

# Override split lists when preparing a partial local dataset.
REFCOCO_SPLITS ?= train val testA testB
REFCOCOG_SPLITS ?= train val

# Override model/eval outputs as needed for a specific run.
LORA_RANK ?= 8
FINETUNE_OUTPUT_DIR ?= checkpoints/qlora_r$(LORA_RANK)
FINETUNE_CONTEXT_OUTPUT_DIR ?= checkpoints/qlora_context_r$(LORA_RANK)
CONDITION ?= A
PREDICTIONS ?= results/predictions_cond$(CONDITION)_$(DATASET)_$(SPLIT)_$(SUBSET).jsonl
GROUND_TRUTH ?= data/processed/$(DATASET)_$(SPLIT).jsonl
SUBSET_FILE ?= data/splits/$(DATASET)_$(SPLIT)_$(SUBSET).jsonl
MAX_RELATIONS ?=
MAX_RELATIONS_ARG = $(if $(MAX_RELATIONS),--max_relations $(MAX_RELATIONS),)

env:
	conda env create -f environment.yaml

prepare-data:
	@set -e; \
	for dataset in refcoco refcoco+; do \
		for split in $(REFCOCO_SPLITS); do \
			$(PYTHON) data/prepare_refcoco.py --config $(CONFIG) --dataset $$dataset --split $$split; \
		done; \
	done; \
	for split in $(REFCOCOG_SPLITS); do \
		$(PYTHON) data/prepare_refcoco.py --config $(CONFIG) --dataset refcocog --split $$split; \
	done; \
	$(PYTHON) data/join_visual_genome.py --config $(CONFIG)

classify:
	$(PYTHON) data/classify_expressions.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT)

baseline:
	$(PYTHON) prompting/zero_shot_baseline.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET)

prompted:
	$(PYTHON) prompting/relation_prompted.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET) $(MAX_RELATIONS_ARG)

finetune:
	$(PYTHON) finetune/train_qlora.py --config $(CONFIG) --dataset $(DATASET) --split train --lora_rank $(LORA_RANK) --output_dir $(FINETUNE_OUTPUT_DIR)

finetune-context:
	$(PYTHON) finetune/train_qlora_with_context.py --config $(CONFIG) --dataset $(DATASET) --split train --lora_rank $(LORA_RANK) --output_dir $(FINETUNE_CONTEXT_OUTPUT_DIR)

eval:
	$(PYTHON) eval/compute_accuracy_iou.py --config $(CONFIG) --predictions $(PREDICTIONS) --ground_truth $(GROUND_TRUTH) --subset_file $(SUBSET_FILE) --condition $(CONDITION) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET)

results:
	$(PYTHON) eval/build_results_table.py --config $(CONFIG)

test:
	pytest tests/ -q

lint:
	ruff check .
