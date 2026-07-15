"""Train condition D QLoRA adapter with Visual Genome relation context."""

from __future__ import annotations

import argparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import setup_logger  # noqa: E402
from finetune.train_qlora import (  # noqa: E402
    InputFileError,
    RefCocoBoxDataset,
    ensure_file,
    load_jsonl,
    parse_args,
    run,
    safe_int,
)

LOGGER = setup_logger("train_qlora_with_context")


def load_relations_by_coco_id(path: Path) -> dict[int, list[dict]]:
    mapping: dict[int, list[dict]] = {}
    for row in load_jsonl(path):
        coco_id = safe_int(row.get("coco_id"))
        relations = row.get("relations")
        if coco_id is not None and isinstance(relations, list):
            mapping[coco_id] = [
                relation for relation in relations if isinstance(relation, dict)
            ]
    return mapping


class RefCocoContextBoxDataset(RefCocoBoxDataset):
    def __init__(
        self,
        jsonl_path: Path,
        coco_images_dir: Path,
        max_long_edge: int | None,
        relations_by_coco_id: dict[int, list[dict]],
        max_relations: int,
    ):
        super().__init__(jsonl_path, coco_images_dir, max_long_edge)
        self.relations_by_coco_id = relations_by_coco_id
        self.max_relations = max_relations

    def relations_for_row(self, row: dict) -> list[dict]:
        coco_id = safe_int(row.get("coco_id"))
        return self.relations_by_coco_id.get(coco_id, [])[: self.max_relations]


def parse_context_args() -> argparse.Namespace:
    context_parser = argparse.ArgumentParser(add_help=False)
    context_parser.add_argument(
        "--max_relations",
        type=int,
        default=10,
        help="Maximum number of Visual Genome relation triplets injected per example.",
    )
    context_args, remaining_args = context_parser.parse_known_args()
    if context_args.max_relations < 0:
        context_parser.error("--max_relations must be non-negative")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining_args]
        args = parse_args(__doc__)
    finally:
        sys.argv = original_argv
    args.max_relations = context_args.max_relations
    return args


def build_context_dataset(
    args: argparse.Namespace,
    config: dict,
    processed_dir: Path,
    coco_images_dir: Path,
    max_long_edge: int | None,
):
    split_path = ensure_file(
        processed_dir / f"{args.dataset}_{args.split}.jsonl",
        "processed RefCOCO JSONL",
    )
    relations_path = ensure_file(
        processed_dir / "vg_relations_by_coco_id.jsonl",
        "Visual Genome relation mapping",
    )
    relations_by_coco_id = load_relations_by_coco_id(relations_path)
    LOGGER.info("Loaded relations for %d COCO ids", len(relations_by_coco_id))
    return RefCocoContextBoxDataset(
        split_path,
        coco_images_dir,
        max_long_edge,
        relations_by_coco_id,
        args.max_relations,
    )


def main() -> int:
    args = parse_context_args()
    LOGGER.info("Effective max_relations per example: %d", args.max_relations)
    return run(args, dataset_builder=build_context_dataset, logger=LOGGER)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
