"""Train condition C QLoRA adapter without relation context."""

from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from common.utils import load_config, resize_long_edge, set_seed, setup_logger  # noqa: E402
from prompting.vlm_utils import build_prompt, load_vlm  # noqa: E402

LOGGER = setup_logger("train_qlora")


class InputFileError(Exception):
    """Raised when an expected local input file is missing or malformed."""


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise InputFileError(f"Missing required {label}: expected '{path}'.")
    return path


def require_config_path(config: dict, key: str) -> Path:
    value = config.get("paths", {}).get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputFileError(f"Missing config entry 'paths.{key}'.")
    return resolve_repo_path(value)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise InputFileError(
                    f"Invalid JSON on line {line_number} of '{path}': {exc}"
                ) from exc
    return rows


def safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def config_default_lora(config_path_str: str) -> tuple[int, bool]:
    path = resolve_repo_path(config_path_str)
    if not path.is_file():
        return 8, True
    config = load_config(str(path))
    lora_config = config.get("lora", {})
    rank = safe_int(lora_config.get("rank")) or 8
    freeze = bool(lora_config.get("freeze_vision_tower", True))
    return rank, freeze


def parse_args(description: str | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="configs/config.yaml")
    known_args, _ = bootstrap.parse_known_args()
    default_rank, default_freeze = config_default_lora(known_args.config)

    parser = argparse.ArgumentParser(
        description=description or __doc__, parents=[bootstrap]
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--split", required=True, help="Training split to read from data/processed."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="Override lora.rank for rank-sweep ablations.",
    )
    parser.add_argument(
        "--freeze_vision_tower",
        action=argparse.BooleanOptionalAction,
        default=default_freeze,
        help="Freeze parameters whose names contain 'vision' or 'visual'.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Stable directory for periodic checkpoints and the final LoRA adapter. "
            "Reuse the same value when resuming a run."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save a resumable Trainer checkpoint every N update steps (default: 50).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Keep at most this many Trainer checkpoints (default: 3).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        default=None,
        const=None,
        metavar="PATH|auto",
        help=(
            "Checkpoint directory to resume from, or 'auto' to use the latest "
            "checkpoint in --output_dir. Omit this option (or provide it without "
            "a value) to start fresh."
        ),
    )
    args = parser.parse_args()

    rank = args.lora_rank if args.lora_rank is not None else default_rank
    if rank <= 0:
        parser.error("--lora_rank must be positive")
    if args.save_steps <= 0:
        parser.error("--save_steps must be positive")
    if args.save_total_limit <= 0:
        parser.error("--save_total_limit must be positive")
    if args.output_dir is None:
        args.output_dir = f"checkpoints/qlora_r{rank}"
    return args


def resolve_image_path(row: dict, coco_images_dir: Path) -> Path:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return resolve_repo_path(image_path)
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return coco_images_dir / file_name
    raise KeyError("row has neither image_path nor file_name")


def scaled_bbox(
    bbox: Any, original_size: tuple[int, int], resized_size: tuple[int, int]
) -> list[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    orig_width, orig_height = original_size
    new_width, new_height = resized_size
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    x1, y1, x2, y2 = (float(value) for value in bbox)
    values = [
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    ]
    return [
        max(0, min(new_width, values[0])),
        max(0, min(new_height, values[1])),
        max(0, min(new_width, values[2])),
        max(0, min(new_height, values[3])),
    ]


def bbox_target_string(bbox: list[int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"<box>({x1},{y1}),({x2},{y2})</box>"


class RefCocoBoxDataset(Dataset):
    def __init__(
        self, jsonl_path: Path, coco_images_dir: Path, max_long_edge: int | None
    ):
        self.rows = load_jsonl(jsonl_path)
        self.coco_images_dir = coco_images_dir
        self.max_long_edge = max_long_edge

    def __len__(self) -> int:
        return len(self.rows)

    def relations_for_row(self, row: dict) -> list[dict] | None:
        return None

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        expression = row.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Row {index} is missing a valid expression")

        image_path = resolve_image_path(row, self.coco_images_dir)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        if isinstance(self.max_long_edge, int) and self.max_long_edge > 0:
            image = resize_long_edge(image, self.max_long_edge)

        bbox = scaled_bbox(row.get("bbox_xyxy"), original_size, image.size)
        return {
            "image": image,
            "prompt": build_prompt(expression.strip(), self.relations_for_row(row)),
            "target": bbox_target_string(bbox),
        }


class QwenVLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)

    @staticmethod
    def _messages(feature: dict, include_target: bool) -> list[dict]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": feature["image"]},
                    {"type": "text", "text": feature["prompt"]},
                ],
            }
        ]
        if include_target:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": feature["target"]}],
                }
            )
        return messages

    def _apply_processor(
        self, messages_batch: list[list[dict]], texts: list[str], images: list
    ):
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs = []
            video_inputs = []
            for messages in messages_batch:
                sample_images, sample_videos = process_vision_info(messages)
                if sample_images:
                    image_inputs.extend(sample_images)
                if sample_videos:
                    video_inputs.extend(sample_videos)
            return self.processor(
                text=texts,
                images=image_inputs or None,
                videos=video_inputs or None,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            return self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

    def _prompt_length(self, feature: dict) -> int:
        messages = self._messages(feature, include_target=False)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._apply_processor([messages], [text], [feature["image"]])
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return int(inputs["input_ids"].shape[1])
        return int(attention_mask[0].sum().item())

    def __call__(self, features: list[dict]) -> dict:
        messages_batch = [
            self._messages(feature, include_target=True) for feature in features
        ]
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in messages_batch
        ]
        images = [feature["image"] for feature in features]
        inputs = self._apply_processor(messages_batch, texts, images)

        labels = inputs["input_ids"].clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        attention_mask = inputs.get("attention_mask")
        prompt_lengths = [self._prompt_length(feature) for feature in features]
        for index, prompt_length in enumerate(prompt_lengths):
            if attention_mask is not None:
                non_padding = torch.nonzero(
                    attention_mask[index], as_tuple=False
                ).flatten()
                start = int(non_padding[0].item()) if non_padding.numel() else 0
            else:
                start = 0
            stop = min(start + prompt_length, labels.shape[1])
            labels[index, start:stop] = -100

        inputs["labels"] = labels
        return inputs


def max_image_long_edge(config: dict) -> int | None:
    value = config.get("model", {}).get("max_image_long_edge")
    return value if isinstance(value, int) and value > 0 else None


def force_4bit_config(config: dict) -> dict:
    updated = dict(config)
    model_config = dict(updated.get("model", {}))
    model_config["load_in_4bit"] = True
    updated["model"] = model_config
    return updated


def lora_rank(args: argparse.Namespace, config: dict) -> int:
    return (
        args.lora_rank
        if args.lora_rank is not None
        else int(config.get("lora", {}).get("rank", 8))
    )


def lora_alpha(config: dict) -> int:
    return int(config.get("lora", {}).get("alpha", 16))


def lora_target_modules(config: dict) -> list[str]:
    modules = config.get("lora", {}).get("target_modules")
    if not isinstance(modules, list) or not all(
        isinstance(module, str) for module in modules
    ):
        raise InputFileError("Missing or invalid config entry 'lora.target_modules'.")
    return modules


def freeze_vision_tower(model) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        if "vision" in name.lower() or "visual" in name.lower():
            parameter.requires_grad = False
            frozen += 1
    return frozen


def build_peft_model(model, args: argparse.Namespace, config: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_rank(args, config),
        lora_alpha=lora_alpha(config),
        target_modules=lora_target_modules(config),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if args.freeze_vision_tower:
        frozen = freeze_vision_tower(model)
        LOGGER.info("Froze %d vision/visual parameters", frozen)
    model.print_trainable_parameters()
    return model


def build_dataset(
    args: argparse.Namespace,
    config: dict,
    processed_dir: Path,
    coco_images_dir: Path,
    max_long_edge: int | None,
) -> Dataset:
    split_path = ensure_file(
        processed_dir / f"{args.dataset}_{args.split}.jsonl",
        "processed RefCOCO JSONL",
    )
    return RefCocoBoxDataset(split_path, coco_images_dir, max_long_edge)


def trainer_args(args: argparse.Namespace, config: dict, output_dir: Path):
    from transformers import TrainingArguments

    training_config = config.get("training", {})
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_config.get("batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("grad_accum_steps", 1)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        num_train_epochs=float(training_config.get("epochs", 1)),
        seed=int(training_config.get("seed", 42)),
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        bf16=bf16,
        fp16=fp16,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
    )


def checkpoint_step(checkpoint_path: Path) -> int | None:
    trainer_state_path = checkpoint_path / "trainer_state.json"
    try:
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        trainer_state = {}
    step = safe_int(trainer_state.get("global_step"))
    if step is not None:
        return step

    prefix = "checkpoint-"
    if checkpoint_path.name.startswith(prefix):
        return safe_int(checkpoint_path.name.removeprefix(prefix))
    return None


def resolve_resume_checkpoint(
    resume_from_checkpoint: str | None,
    output_dir: Path,
    logger=LOGGER,
) -> str | None:
    if resume_from_checkpoint is None:
        logger.info("Checkpoint resume mode: fresh start")
        return None

    if resume_from_checkpoint == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        resolved = get_last_checkpoint(str(output_dir))
        if resolved is None:
            logger.info(
                "Checkpoint resume mode: auto; no checkpoint found in %s; "
                "starting fresh",
                output_dir,
            )
            return None
        checkpoint_path = Path(resolved).resolve()
        mode = "auto-detected"
    else:
        checkpoint_path = resolve_repo_path(resume_from_checkpoint).resolve()
        if not checkpoint_path.is_dir():
            raise InputFileError(
                "Resume checkpoint directory does not exist: "
                f"expected '{checkpoint_path}'."
            )
        mode = "explicit"

    step = checkpoint_step(checkpoint_path)
    step_text = f" at step {step}" if step is not None else ""
    logger.info(
        "Checkpoint resume mode: %s; resuming from %s%s",
        mode,
        checkpoint_path,
        step_text,
    )
    return str(checkpoint_path)


def append_run_log(
    results_dir: Path,
    rank: int,
    freeze: bool,
    peak_mem_mb: float,
    train_seconds: float,
    output_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "finetune_run_log.csv"
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "freeze_vision_tower",
                "peak_mem_mb",
                "train_seconds",
                "output_dir",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "rank": rank,
                "freeze_vision_tower": freeze,
                "peak_mem_mb": f"{peak_mem_mb:.2f}",
                "train_seconds": f"{train_seconds:.2f}",
                "output_dir": str(output_dir),
            }
        )


def run(
    args: argparse.Namespace,
    dataset_builder=build_dataset,
    logger=LOGGER,
) -> int:
    config_path = ensure_file(resolve_repo_path(args.config), "config file")
    config = load_config(str(config_path))
    set_seed(int(config.get("training", {}).get("seed", 42)))

    processed_dir = require_config_path(config, "processed_dir")
    coco_images_dir = require_config_path(config, "coco_images_dir")
    results_dir = require_config_path(config, "results_dir")
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        output_dir,
        logger,
    )

    train_dataset = dataset_builder(
        args,
        config,
        processed_dir,
        coco_images_dir,
        max_image_long_edge(config),
    )
    logger.info("Loaded %d training examples", len(train_dataset))

    model, processor = load_vlm(force_4bit_config(config))
    model = build_peft_model(model, args, config)

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=trainer_args(args, config, output_dir),
        train_dataset=train_dataset,
        data_collator=QwenVLDataCollator(processor),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.perf_counter() - started
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )

    model.save_pretrained(output_dir)
    append_run_log(
        results_dir,
        lora_rank(args, config),
        bool(args.freeze_vision_tower),
        peak_mem_mb,
        train_seconds,
        output_dir,
    )
    logger.info("Saved LoRA adapter to %s", output_dir)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
