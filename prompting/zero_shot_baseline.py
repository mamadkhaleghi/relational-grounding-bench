"""Run condition A zero-shot VLM prompting on a classified split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import load_config, parse_bbox_from_text, setup_logger  # noqa: E402
from prompting.vlm_utils import build_prompt, load_vlm, run_inference  # noqa: E402

LOGGER = setup_logger("zero_shot_baseline")


class InputFileError(Exception):
    """Raised when an expected local input file is missing."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset", required=True, choices=["relational", "attribute"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def require_config_path(config: dict, key: str) -> Path:
    value = config.get("paths", {}).get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputFileError(f"Missing config entry 'paths.{key}'.")
    return resolve_repo_path(value)


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise InputFileError(f"Missing required {label}: expected '{path}'.")
    return path


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


def resolve_image_path(row: dict, coco_images_dir: Path) -> Path:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return resolve_repo_path(image_path)
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return coco_images_dir / file_name
    raise KeyError("row has neither image_path nor file_name")


def default_output_path(args: argparse.Namespace, results_dir: Path) -> Path:
    if args.output:
        return resolve_repo_path(args.output)
    return (
        results_dir
        / f"predictions_condA_{args.dataset}_{args.split}_{args.subset}.jsonl"
    )


def write_prediction(handle, row: dict, predicted_bbox, raw_output: str) -> None:
    output_row = {
        "ref_id": row.get("ref_id"),
        "expression": row.get("expression"),
        "predicted_bbox": list(predicted_bbox) if predicted_bbox is not None else None,
        "raw_output": raw_output,
    }
    handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    handle.flush()


def run(args: argparse.Namespace) -> int:
    config_path = ensure_file(resolve_repo_path(args.config), "config file")
    config = load_config(str(config_path))
    splits_dir = require_config_path(config, "splits_dir")
    results_dir = require_config_path(config, "results_dir")
    coco_images_dir = require_config_path(config, "coco_images_dir")

    split_path = ensure_file(
        splits_dir / f"{args.dataset}_{args.split}_{args.subset}.jsonl",
        "classified split JSONL",
    )
    output_path = default_output_path(args, results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(split_path)
    LOGGER.info("Loaded %d examples from %s", len(rows), split_path)
    model, processor = load_vlm(config)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            expression = row.get("expression")
            try:
                if not isinstance(expression, str) or not expression.strip():
                    raise ValueError("missing expression")
                image_path = resolve_image_path(row, coco_images_dir)
                prompt = build_prompt(expression, relations=None)
                raw_output = run_inference(model, processor, str(image_path), prompt)
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
