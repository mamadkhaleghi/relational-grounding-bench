"""Run condition C/D inference with a saved LoRA adapter."""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from common.utils import load_config, parse_bbox_from_text, setup_logger  # noqa: E402
from prompting.relation_prompted import (  # noqa: E402
    InputFileError,
    capped_relations,
    load_jsonl,
    load_relations_by_coco_id,
    max_relations_label,
    safe_int,
    write_prediction,
)
from prompting.vlm_utils import build_prompt, load_vlm, run_inference  # noqa: E402

LOGGER = setup_logger("finetuned_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--subset", required=True, choices=["relational", "positional", "attribute"]
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--condition", required=True, choices=["C", "D"])
    parser.add_argument("--max_relations", type=int, default=None)
    args = parser.parse_args()
    if args.max_relations is not None and args.max_relations < 0:
        parser.error("--max_relations must be non-negative")
    return args


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def require_config_path(config: dict, key: str) -> Path:
    value = config.get("paths", {}).get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputFileError(f"Missing config entry 'paths.{key}'.")
    return resolve_repo_path(value)


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise InputFileError(f"Missing required {label}: expected '{path}'.")
    return path


def ensure_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise InputFileError(f"Missing required {label}: expected '{path}'.")
    return path


def resolve_image_path(row: dict, coco_images_dir: Path) -> Path:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return resolve_repo_path(image_path)
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return coco_images_dir / file_name
    raise KeyError("row has neither image_path nor file_name")


def default_output_path(args: argparse.Namespace, results_dir: Path) -> Path:
    base_name = (
        f"predictions_cond{args.condition}_"
        f"{args.dataset}_{args.split}_{args.subset}"
    )
    if args.condition == "D":
        base_name += f"_{max_relations_label(args.max_relations)}"
    return results_dir / f"{base_name}.jsonl"


def run(args: argparse.Namespace) -> int:
    config_path = ensure_file(resolve_repo_path(args.config), "config file")
    adapter_dir = ensure_directory(
        resolve_repo_path(args.adapter_dir), "LoRA adapter directory"
    )
    config = load_config(str(config_path))
    splits_dir = require_config_path(config, "splits_dir")
    processed_dir = require_config_path(config, "processed_dir")
    results_dir = require_config_path(config, "results_dir")
    coco_images_dir = require_config_path(config, "coco_images_dir")

    split_path = ensure_file(
        splits_dir / f"{args.dataset}_{args.split}_{args.subset}.jsonl",
        "classified split JSONL",
    )
    relations_by_coco_id = None
    if args.condition == "D":
        relations_path = ensure_file(
            processed_dir / "vg_relations_by_coco_id.jsonl",
            "Visual Genome relation mapping",
        )
        relations_by_coco_id = load_relations_by_coco_id(relations_path)
        LOGGER.info("Loaded relations for %d COCO ids", len(relations_by_coco_id))

    output_path = default_output_path(args, results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(split_path)
    LOGGER.info("Loaded %d examples from %s", len(rows), split_path)

    base_model, processor = load_vlm(config)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            try:
                expression = row.get("expression")
                if not isinstance(expression, str) or not expression.strip():
                    raise ValueError("missing expression")
                image_path = resolve_image_path(row, coco_images_dir)
                relations = None
                if relations_by_coco_id is not None:
                    relations = capped_relations(
                        relations_by_coco_id.get(safe_int(row.get("coco_id")), []),
                        args.max_relations,
                    )
                prompt = build_prompt(expression, relations=relations)
                raw_output = run_inference(
                    model, processor, str(image_path), prompt
                )
                predicted_bbox = parse_bbox_from_text(raw_output)
                write_prediction(handle, row, predicted_bbox, raw_output)
            except Exception as exc:
                LOGGER.exception(
                    "Failed example %s at row %d: %s",
                    row.get("ref_id"),
                    index,
                    exc,
                )
                write_prediction(handle, row, None, "")

            if index % 50 == 0:
                LOGGER.info("Processed %d / %d examples", index, len(rows))

    LOGGER.info("Wrote predictions to %s", output_path)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
