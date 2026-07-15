"""Compute accuracy at an IoU threshold for a prediction JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
import re
from typing import Any

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import compute_iou, load_config  # noqa: E402

TABLE_COLUMNS = [
    "condition",
    "dataset",
    "split",
    "subset",
    "n_examples",
    "accuracy_at_iou50",
]
SUBSETS = ["relational", "positional", "attribute"]


class InputFileError(Exception):
    """Raised when an expected input file is missing or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Condition prediction JSONL to score.",
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Processed RefCOCO JSONL containing bbox_xyxy ground-truth boxes.",
    )
    parser.add_argument(
        "--subset_file",
        default=None,
        help="Optional split JSONL whose ref_ids restrict the scored predictions.",
    )
    parser.add_argument("--condition", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--subset", default=None, choices=SUBSETS)
    return parser.parse_args()


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
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise InputFileError(
                    f"Invalid JSON on line {line_number} of '{path}': {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise InputFileError(
                    f"Expected JSON object on line {line_number} of '{path}'."
                )
            rows.append(row)
    return rows


def ref_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def bbox_from_row(row: dict, keys: tuple[str, ...]) -> list[float] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                return None
    return None


def load_ground_truth_by_ref_id(path: Path) -> dict[str, list[float]]:
    mapping: dict[str, list[float]] = {}
    for row in load_jsonl(path):
        key = ref_key(row.get("ref_id"))
        bbox = bbox_from_row(row, ("bbox_xyxy", "bbox"))
        if key is None or bbox is None:
            continue
        mapping.setdefault(key, bbox)
    if not mapping:
        raise InputFileError(f"No usable ground-truth boxes found in '{path}'.")
    return mapping


def load_ref_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ref_ids = {
        key for row in load_jsonl(path) for key in [ref_key(row.get("ref_id"))] if key
    }
    if not ref_ids:
        raise InputFileError(f"No usable ref_id values found in subset file '{path}'.")
    return ref_ids


def infer_metadata(predictions_path: Path) -> dict[str, str | None]:
    stem = predictions_path.stem
    parts = stem.split("_")
    metadata: dict[str, str | None] = {
        "condition": None,
        "dataset": None,
        "split": None,
        "subset": None,
    }

    for index, part in enumerate(parts):
        match = re.fullmatch(r"cond([A-D])", part, re.IGNORECASE)
        if match:
            metadata["condition"] = match.group(1).upper()
            if index + 1 < len(parts):
                metadata["dataset"] = parts[index + 1]
            if index + 2 < len(parts):
                metadata["split"] = parts[index + 2]
            if index + 3 < len(parts) and parts[index + 3] in SUBSETS:
                metadata["subset"] = parts[index + 3]
            break

    if metadata["subset"] is None:
        for part in parts:
            if part in SUBSETS:
                metadata["subset"] = part
                break

    if metadata["condition"] is None:
        match = re.search(r"(?:^|[_-])([A-D])(?:[_-]|$)", stem, re.IGNORECASE)
        if match:
            metadata["condition"] = match.group(1).upper()

    return metadata


def metadata_value(
    overrides: argparse.Namespace,
    inferred: dict[str, str | None],
    field: str,
) -> str:
    value = getattr(overrides, field)
    if value is None:
        value = inferred.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputFileError(
            f"Could not infer {field} from the predictions filename; pass --{field}."
        )
    value = value.strip()
    if field == "condition":
        value = value.upper()
        if value.startswith("COND"):
            value = value[4:]
        if value not in {"A", "B", "C", "D"}:
            raise InputFileError(
                "Condition must be one of A, B, C, or D; "
                f"got '{getattr(overrides, field) or inferred.get(field)}'."
            )
    return value


def score_predictions(
    predictions: list[dict],
    ground_truth: dict[str, list[float]],
    allowed_ref_ids: set[str] | None,
    threshold: float,
) -> tuple[int, int]:
    total = 0
    hits = 0
    missing_ref_ids = []

    for row in predictions:
        key = ref_key(row.get("ref_id"))
        if key is None:
            continue
        if allowed_ref_ids is not None and key not in allowed_ref_ids:
            continue
        gt_bbox = ground_truth.get(key)
        if gt_bbox is None:
            missing_ref_ids.append(key)
            continue

        total += 1
        predicted_bbox = bbox_from_row(
            row, ("predicted_bbox", "bbox", "prediction_bbox")
        )
        if predicted_bbox is None:
            continue
        if compute_iou(predicted_bbox, gt_bbox) >= threshold:
            hits += 1

    if missing_ref_ids:
        sample = ", ".join(missing_ref_ids[:5])
        raise InputFileError(
            f"{len(missing_ref_ids)} prediction rows had no matching ground truth "
            f"ref_id. Sample: {sample}"
        )
    return hits, total


def append_accuracy_row(results_dir: Path, row: dict[str, str | int]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    table_path = results_dir / "accuracy_table.csv"
    write_header = not table_path.exists() or table_path.stat().st_size == 0

    with table_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return table_path


def run(args: argparse.Namespace) -> int:
    config_path = ensure_file(resolve_repo_path(args.config), "config file")
    predictions_path = ensure_file(
        resolve_repo_path(args.predictions), "predictions JSONL"
    )
    ground_truth_path = ensure_file(
        resolve_repo_path(args.ground_truth), "ground-truth JSONL"
    )
    subset_path = (
        ensure_file(resolve_repo_path(args.subset_file), "subset JSONL")
        if args.subset_file
        else None
    )

    config = load_config(str(config_path))
    results_dir = require_config_path(config, "results_dir")
    threshold = float(config.get("eval", {}).get("iou_threshold", 0.5))
    inferred = infer_metadata(predictions_path)

    condition = metadata_value(args, inferred, "condition")
    dataset = metadata_value(args, inferred, "dataset")
    split = metadata_value(args, inferred, "split")
    subset = metadata_value(args, inferred, "subset")

    predictions = load_jsonl(predictions_path)
    ground_truth = load_ground_truth_by_ref_id(ground_truth_path)
    allowed_ref_ids = load_ref_id_filter(subset_path)
    hits, total = score_predictions(
        predictions, ground_truth, allowed_ref_ids, threshold
    )
    accuracy = hits / total if total else 0.0

    row = {
        "condition": condition,
        "dataset": dataset,
        "split": split,
        "subset": subset,
        "n_examples": total,
        "accuracy_at_iou50": f"{accuracy:.4f}",
    }
    table_path = append_accuracy_row(results_dir, row)

    print(
        "condition={condition} dataset={dataset} split={split} subset={subset} "
        "n_examples={total} hits={hits} accuracy_at_iou50={accuracy:.4f} "
        "threshold={threshold:.2f} appended={table_path}".format(
            condition=condition,
            dataset=dataset,
            split=split,
            subset=subset,
            total=total,
            hits=hits,
            accuracy=accuracy,
            threshold=threshold,
            table_path=table_path,
        )
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
