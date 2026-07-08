"""Train condition D QLoRA adapter with Visual Genome relation context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    ):
        super().__init__(jsonl_path, coco_images_dir, max_long_edge)
        self.relations_by_coco_id = relations_by_coco_id

    def relations_for_row(self, row: dict) -> list[dict]:
        coco_id = safe_int(row.get("coco_id"))
        return self.relations_by_coco_id.get(coco_id, [])


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
    )


def main() -> int:
    return run(
        parse_args(__doc__), dataset_builder=build_context_dataset, logger=LOGGER
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
