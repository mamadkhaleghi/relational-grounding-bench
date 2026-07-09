"""Join Visual Genome relations onto COCO image ids."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import load_config, setup_logger

LOGGER = setup_logger("join_visual_genome")


class InputFileError(Exception):
    """Raised when an expected local input file is missing."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def object_name(obj: dict, fallback: dict | None = None):
    for candidate in (obj, fallback or {}):
        name = candidate.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        names = candidate.get("names")
        if isinstance(names, list):
            for value in names:
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def object_box(obj: dict, fallback: dict | None = None):
    for candidate in (obj, fallback or {}):
        if all(key in candidate for key in ("x", "y", "w", "h")):
            x = float(candidate["x"])
            y = float(candidate["y"])
            w = float(candidate["w"])
            h = float(candidate["h"])
            return [x, y, x + w, y + h]
    return None


def safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_mapping(config_path_str: str) -> int:
    config = load_config_paths(config_path_str)
    vg_dir = require_config_path(config, "visual_genome_dir")
    processed_dir = require_config_path(config, "processed_dir")

    image_data_path = ensure_file(vg_dir / "image_data.json", "Visual Genome image metadata")
    objects_path = ensure_file(vg_dir / "objects.json", "Visual Genome objects data")
    relationships_path = ensure_file(
        vg_dir / "relationships.json",
        "Visual Genome relationships data",
    )

    image_data = load_json(image_data_path)
    objects_data = load_json(objects_path)
    relationships_data = load_json(relationships_path)

    total_vg_images = len(image_data)
    image_id_to_coco_id = {}
    coco_relations = defaultdict(list)

    for image_record in image_data:
        image_id = safe_int(image_record.get("image_id"))
        coco_id = safe_int(image_record.get("coco_id"))
        if image_id is None or coco_id is None:
            continue
        image_id_to_coco_id[image_id] = coco_id
        coco_relations[coco_id]

    objects_by_image = {}
    for image_record in objects_data:
        image_id = safe_int(image_record.get("image_id"))
        if image_id is None:
            continue
        objects_by_image[image_id] = {
            safe_int(obj["object_id"]): obj
            for obj in image_record.get("objects", [])
            if "object_id" in obj and safe_int(obj["object_id"]) is not None
        }

    for image_record in relationships_data:
        image_id = safe_int(image_record.get("image_id"))
        if image_id is None:
            continue
        coco_id = image_id_to_coco_id.get(image_id)
        if coco_id is None:
            continue

        object_index = objects_by_image.get(image_id, {})
        for relation in image_record.get("relationships", []):
            subject = relation.get("subject") or {}
            obj = relation.get("object") or {}
            subject_fallback = object_index.get(safe_int(subject.get("object_id")))
            object_fallback = object_index.get(safe_int(obj.get("object_id")))

            coco_relations[coco_id].append(
                {
                    "subject": object_name(subject, subject_fallback),
                    "predicate": relation.get("predicate"),
                    "object": object_name(obj, object_fallback),
                    "subject_box": object_box(subject, subject_fallback),
                    "object_box": object_box(obj, object_fallback),
                }
            )

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "vg_relations_by_coco_id.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for coco_id in sorted(coco_relations):
            handle.write(
                json.dumps(
                    {"coco_id": coco_id, "relations": coco_relations[coco_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    LOGGER.info("Total VG images: %d", total_vg_images)
    LOGGER.info("VG images with a coco_id: %d", len(image_id_to_coco_id))
    LOGGER.info("Resulting coco_id mapping size: %d", len(coco_relations))
    LOGGER.info("Wrote Visual Genome relations to %s", output_path)
    return 0


def main() -> int:
    args = parse_args()
    return build_mapping(args.config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
