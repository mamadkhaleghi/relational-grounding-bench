"""Build README-ready Markdown result tables from local CSV logs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import load_config  # noqa: E402

CONDITIONS = ["A", "B", "C", "D"]
SUBSETS = ["relational", "attribute"]


class InputFileError(Exception):
    """Raised when the config file or required config values are missing."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
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


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def read_csv_rows(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        warn(f"missing {label}; skipping '{path}'.")
        return []
    if path.stat().st_size == 0:
        warn(f"empty {label}; skipping '{path}'.")
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            warn(f"{label} has no header; skipping '{path}'.")
            return []
        rows = [
            row
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    if not rows:
        warn(f"{label} has no data rows; skipping '{path}'.")
    return rows


def normalize_condition(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().upper()
    if text.startswith("COND"):
        text = text[4:]
    return text if text in CONDITIONS else None


def normalize_subset(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    return text if text in SUBSETS else None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def format_number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def accuracy_value(row: dict) -> float | None:
    for key in ("accuracy_at_iou50", "accuracy"):
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def build_accuracy_table(rows: list[dict]) -> str:
    latest: dict[tuple[str, str], float | None] = {}
    for row in rows:
        condition = normalize_condition(row.get("condition"))
        subset = normalize_subset(row.get("subset"))
        if condition is None or subset is None:
            continue
        latest[(condition, subset)] = accuracy_value(row)

    if not latest:
        return "_Accuracy results unavailable._"

    lines = [
        "| condition | relational | attribute | gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        relational = latest.get((condition, "relational"))
        attribute = latest.get((condition, "attribute"))
        gap = (
            attribute - relational
            if relational is not None and attribute is not None
            else None
        )
        lines.append(
            "| {condition} | {relational} | {attribute} | {gap} |".format(
                condition=condition,
                relational=format_metric(relational),
                attribute=format_metric(attribute),
                gap=format_metric(gap),
            )
        )
    return "\n".join(lines)


def finetune_accuracy(row: dict) -> float | None:
    for key in ("accuracy", "accuracy_at_iou50", "eval_accuracy"):
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def build_finetune_table(rows: list[dict]) -> str:
    latest_by_rank: dict[int, dict] = {}
    for row in rows:
        rank = parse_int(row.get("rank"))
        if rank is None:
            continue
        latest_by_rank[rank] = row

    if not latest_by_rank:
        return "_LoRA rank-sweep results unavailable._"

    lines = [
        "| rank | accuracy | peak_mem_mb | train_seconds |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for rank in sorted(latest_by_rank):
        row = latest_by_rank[rank]
        lines.append(
            "| {rank} | {accuracy} | {peak_mem} | {seconds} |".format(
                rank=rank,
                accuracy=format_metric(finetune_accuracy(row)),
                peak_mem=format_number(parse_float(row.get("peak_mem_mb"))),
                seconds=format_number(parse_float(row.get("train_seconds"))),
            )
        )
    return "\n".join(lines)


def write_results_markdown(
    output_path: Path,
    accuracy_table: str,
    finetune_table: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "## Accuracy by Condition",
            "",
            accuracy_table,
            "",
            "## LoRA Rank-Sweep Ablation",
            "",
            finetune_table,
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config_path = ensure_file(resolve_repo_path(args.config), "config file")
    config = load_config(str(config_path))
    results_dir = require_config_path(config, "results_dir")

    accuracy_rows = read_csv_rows(
        results_dir / "accuracy_table.csv", "accuracy table"
    )
    finetune_rows = read_csv_rows(
        results_dir / "finetune_run_log.csv", "fine-tune log"
    )

    output_path = results_dir / "results_table.md"
    write_results_markdown(
        output_path,
        build_accuracy_table(accuracy_rows),
        build_finetune_table(finetune_rows),
    )
    print(f"Wrote results tables to {output_path}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
