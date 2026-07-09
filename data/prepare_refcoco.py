"""Prepare RefCOCO family expressions as JSONL rows."""

from __future__ import annotations

import argparse
import json
import pickle
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import load_config, setup_logger

LOGGER = setup_logger("prepare_refcoco")
COCO_ID_PATTERN = re.compile(r"_(\d+)\.")
VALID_SPLITS = {
    "refcoco": {"train", "val", "testA", "testB"},
    "refcoco+": {"train", "val", "testA", "testB"},
    "refcocog": {"train", "val"},
}


class InputFileError(Exception):
    """Raised when an expected local input file is missing or ambiguous."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(VALID_SPLITS),
        help="RefCOCO-family dataset to export.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "testA", "testB"],
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config YAML.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise InputFileError(
            f"Missing required {label}: expected local file at '{path}'. "
            "Download or place the dataset file there and rerun."
        )
    return path


def load_config_paths(config_path_str: str) -> dict:
    config_path = resolve_repo_path(config_path_str)
    ensure_file(config_path, "config file")
    config = load_config(str(config_path))
    config["_config_path"] = str(config_path)
    return config


def require_config_path(config: dict, key: str) -> Path:
    paths = config.get("paths", {})
    value = paths.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputFileError(
            f"Missing config entry 'paths.{key}': expected it in '{config.get('_config_path')}'."
        )
    return resolve_repo_path(value)


def find_refs_pickle(dataset_dir: Path) -> Path:
    matches = sorted(dataset_dir.glob("refs(*).p"))
    if not matches:
        raise InputFileError(
            f"Missing required refs pickle: expected a file matching "
            f"'{dataset_dir / 'refs(*).p'}'. Download or place it there and rerun."
        )
    if len(matches) > 1:
        raise InputFileError(
            f"Expected exactly one refs pickle in '{dataset_dir}', found {len(matches)}: "
            f"{', '.join(path.name for path in matches)}. Keep the intended file only and rerun."
        )
    return matches[0]


def load_refs(path: Path):
    with path.open("rb") as handle:
        refs = pickle.load(handle, encoding="latin1")
    return refs["refs"] if isinstance(refs, dict) and "refs" in refs else refs


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_expression(sentence: dict) -> str | None:
    for key in ("sent", "raw"):
        value = sentence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_coco_id(file_name: str) -> int | None:
    match = COCO_ID_PATTERN.search(file_name)
    return int(match.group(1)) if match else None


def xywh_to_xyxy(bbox) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w, h = (float(value) for value in bbox)
    return [x, y, x + w, y + h]


def safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def export_split(dataset: str, split: str, config_path_str: str) -> int:
    if split not in VALID_SPLITS[dataset]:
        LOGGER.warning(
            "Split '%s' is not available for dataset '%s'; skipping without writing an output file.",
            split,
            dataset,
        )
        return 0

    config = load_config_paths(config_path_str)
    dataset_dir = require_config_path(config, "refcoco_annotations_dir") / dataset
    coco_images_dir = require_config_path(config, "coco_images_dir")
    processed_dir = require_config_path(config, "processed_dir")

    refs_path = ensure_file(find_refs_pickle(dataset_dir), "refs pickle")
    instances_path = ensure_file(dataset_dir / "instances.json", "instances annotation file")

    refs = load_refs(refs_path)
    instances = load_json(instances_path)

    annotations_by_id = {
        annotation_id: annotation
        for annotation in instances.get("annotations", [])
        for annotation_id in [safe_int(annotation.get("id"))]
        if annotation_id is not None
    }
    categories_by_id = {
        category_id: category.get("name")
        for category in instances.get("categories", [])
        for category_id in [safe_int(category.get("id"))]
        if category_id is not None
    }
    images_by_id = {
        image_id: image.get("file_name")
        for image in instances.get("images", [])
        for image_id in [safe_int(image.get("id"))]
        if image_id is not None
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"{dataset}_{split}.jsonl"

    failed_refs = 0
    rows_written = 0
    refs_in_split = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for ref in refs:
            if ref.get("split") != split:
                continue

            refs_in_split += 1
            ref_id = ref.get("ref_id")
            ann_id = ref.get("ann_id")
            ann_id_int = safe_int(ann_id)
            image_id_int = safe_int(ref.get("image_id"))
            annotation = annotations_by_id.get(ann_id_int) if ann_id_int is not None else None
            file_name = ref.get("file_name") or images_by_id.get(image_id_int)
            coco_id = parse_coco_id(file_name) if isinstance(file_name, str) else None
            image_path = coco_images_dir / file_name if isinstance(file_name, str) else None

            if annotation is None or coco_id is None or image_path is None or not image_path.is_file():
                failed_refs += 1
                LOGGER.warning(
                    "Skipping ref_id=%s because image or annotation data could not be resolved "
                    "(ann_id=%s, file_name=%s).",
                    ref_id,
                    ann_id,
                    file_name,
                )
                continue

            bbox_xyxy = xywh_to_xyxy(annotation.get("bbox"))
            category_name = categories_by_id.get(safe_int(annotation.get("category_id")))
            if bbox_xyxy is None or category_name is None:
                failed_refs += 1
                LOGGER.warning(
                    "Skipping ref_id=%s because bbox or category data is missing for ann_id=%s.",
                    ref_id,
                    ann_id,
                )
                continue

            expressions = [
                expression
                for expression in (
                    extract_expression(sentence) for sentence in ref.get("sentences", [])
                )
                if expression
            ]
            if not expressions:
                failed_refs += 1
                LOGGER.warning(
                    "Skipping ref_id=%s because no usable expressions were found in the ref entry.",
                    ref_id,
                )
                continue

            base_row = {
                "ref_id": ref_id,
                "image_id": ref.get("image_id"),
                "coco_id": coco_id,
                "file_name": file_name,
                "bbox_xyxy": bbox_xyxy,
                "category": category_name,
            }
            for expression in expressions:
                row = dict(base_row)
                row["expression"] = expression
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    LOGGER.info(
        "Finished %s/%s: %d refs in split, %d expression rows written, %d refs failed to resolve. "
        "Output: %s",
        dataset,
        split,
        refs_in_split,
        rows_written,
        failed_refs,
        output_path,
    )
    return 0


def main() -> int:
    args = parse_args()
    return export_split(args.dataset, args.split, args.config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
