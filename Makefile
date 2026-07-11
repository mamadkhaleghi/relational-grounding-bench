.PHONY: env prepare-data classify baseline prompted finetune finetune-context finetuned-infer eval results test lint

PYTHON ?= python
CONFIG ?= configs/config.yaml

# Override dataset/split/subset on the command line, e.g. `make baseline DATASET=refcocog SPLIT=val SUBSET=attribute`.
DATASET ?= refcoco
SPLIT ?= val
SUBSET ?= relational

# Override split lists when preparing a partial local dataset.
REFCOCO_SPLITS ?= train val testA testB
REFCOCOG_SPLITS ?= train val test
REFCOCOG_SPLIT_BY ?= umd

# Override inference, training, and evaluation settings for a specific run.
LORA_RANK ?= 8
TRAIN_SPLIT ?= train
MAX_TRAIN_SAMPLES ?= 4000
MAX_STEPS ?=
SAVE_STEPS ?= 50
SAVE_TOTAL_LIMIT ?= 3
RESUME_FROM_CHECKPOINT ?=
FREEZE_VISION_TOWER ?=
FINETUNE_OUTPUT_DIR ?= checkpoints/qlora_r$(LORA_RANK)
FINETUNE_CONTEXT_OUTPUT_DIR ?= checkpoints/qlora_context_r$(LORA_RANK)
CONDITION ?= A
ADAPTER_DIR ?= $(if $(filter D,$(CONDITION)),$(FINETUNE_CONTEXT_OUTPUT_DIR),$(FINETUNE_OUTPUT_DIR))
MAX_RELATIONS ?=
LIMIT_SAMPLES ?=
RELATION_COUNT_LABEL = $(if $(MAX_RELATIONS),$(MAX_RELATIONS),all)
PREDICTION_RELATION_SUFFIX = $(if $(filter B D,$(CONDITION)),_$(RELATION_COUNT_LABEL),)
PREDICTIONS ?= results/predictions_cond$(CONDITION)_$(DATASET)_$(SPLIT)_$(SUBSET)$(PREDICTION_RELATION_SUFFIX).jsonl
GROUND_TRUTH ?= data/processed/$(DATASET)_$(SPLIT).jsonl
SUBSET_FILE ?= data/splits/$(DATASET)_$(SPLIT)_$(SUBSET).jsonl
MAX_RELATIONS_ARG = $(if $(MAX_RELATIONS),--max_relations $(MAX_RELATIONS),)
LIMIT_SAMPLES_ARG = $(if $(LIMIT_SAMPLES),--limit_samples $(LIMIT_SAMPLES),)
MAX_TRAIN_SAMPLES_ARG = $(if $(MAX_TRAIN_SAMPLES),--max_train_samples $(MAX_TRAIN_SAMPLES),)
MAX_STEPS_ARG = $(if $(MAX_STEPS),--max_steps $(MAX_STEPS),)
RESUME_FROM_CHECKPOINT_ARG = $(if $(RESUME_FROM_CHECKPOINT),--resume_from_checkpoint $(RESUME_FROM_CHECKPOINT),)
FREEZE_VISION_TOWER_ARG = $(if $(filter 1 true yes,$(FREEZE_VISION_TOWER)),--freeze_vision_tower,$(if $(filter 0 false no,$(FREEZE_VISION_TOWER)),--no-freeze_vision_tower,))

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
		$(PYTHON) data/prepare_refcoco.py --config $(CONFIG) --dataset refcocog --split $$split --split_by $(REFCOCOG_SPLIT_BY); \
	done; \
	$(PYTHON) data/join_visual_genome.py --config $(CONFIG)

classify:
	$(PYTHON) data/classify_expressions.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT)

baseline:
	$(PYTHON) prompting/zero_shot_baseline.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET) $(LIMIT_SAMPLES_ARG)

prompted:
	$(PYTHON) prompting/relation_prompted.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET) $(MAX_RELATIONS_ARG) $(LIMIT_SAMPLES_ARG)

finetune:
	$(PYTHON) finetune/train_qlora.py --config $(CONFIG) --dataset $(DATASET) --split $(TRAIN_SPLIT) --lora_rank $(LORA_RANK) --output_dir $(FINETUNE_OUTPUT_DIR) --save_steps $(SAVE_STEPS) --save_total_limit $(SAVE_TOTAL_LIMIT) $(MAX_TRAIN_SAMPLES_ARG) $(MAX_STEPS_ARG) $(RESUME_FROM_CHECKPOINT_ARG) $(FREEZE_VISION_TOWER_ARG)

finetune-context:
	$(PYTHON) finetune/train_qlora_with_context.py --config $(CONFIG) --dataset $(DATASET) --split $(TRAIN_SPLIT) --lora_rank $(LORA_RANK) --output_dir $(FINETUNE_CONTEXT_OUTPUT_DIR) --save_steps $(SAVE_STEPS) --save_total_limit $(SAVE_TOTAL_LIMIT) $(MAX_TRAIN_SAMPLES_ARG) $(MAX_STEPS_ARG) $(RESUME_FROM_CHECKPOINT_ARG) $(FREEZE_VISION_TOWER_ARG)

finetuned-infer: CONDITION = C
finetuned-infer:
	$(PYTHON) prompting/finetuned_inference.py --config $(CONFIG) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET) --adapter_dir $(ADAPTER_DIR) --condition $(CONDITION) $(MAX_RELATIONS_ARG)

eval:
	$(PYTHON) eval/compute_accuracy_iou.py --config $(CONFIG) --predictions $(PREDICTIONS) --ground_truth $(GROUND_TRUTH) --subset_file $(SUBSET_FILE) --condition $(CONDITION) --dataset $(DATASET) --split $(SPLIT) --subset $(SUBSET)

results:
	$(PYTHON) eval/build_results_table.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check .
