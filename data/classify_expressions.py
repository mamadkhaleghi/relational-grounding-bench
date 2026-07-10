"""Classify prepared RefCOCO expressions as relational, positional, or attribute.

This version is stricter than the original classifier:
- It avoids treating every spaCy prepositional dependency as relational.
  This prevents common false positives such as "man in red", "bowl of soup",
  and "guy with black shirt".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from functools import lru_cache

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import load_config, setup_logger

LOGGER = setup_logger("classify_expressions")
VALID_SPLITS = {
    "refcoco": {"train", "val", "testA", "testB"},
    "refcoco+": {"train", "val", "testA", "testB"},
    "refcocog": {"train", "val"},
}

POSITION_WORDS = {
    "left",
    "right",
    "center",
    "centre",
    "middle",
    "top",
    "bottom",
    "upper",
    "lower",
    "closest",
    "nearest",
    "farthest",
    "furthest",
    "last",
}
ORDINAL_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}

# A cue must relate to a second, distinct object, matching Visual Genome's
# subject-predicate-object triplets; frame-only positions fall through to "positional".
RELATIONAL_PHRASE_CUES = [
    "left of",
    "right of",
    "next to",
    "near",
    "beside",
    "behind",
    "in front of",
    "front of",
    "above",
    "below",
    "under",
    "beneath",
    "over",
    "between",
    "touching",
    "holding",
    "carrying",
    "riding",
    "throwing to",
    "throwing at",
    "sitting on",
    "standing on",
    "lying on",
    "on top of",
]

# Verbs that generally express interaction between entities.
RELATIONAL_VERBS = {
    "touch",
    "hold",
    "carry",
    "ride",
    "throw",
    "pull",
    "push",
    "grab",
    "hug",
    "kiss",
    "feed",
    "eat",
    "drink",
    "hit",
    "kick",
    "catch",
    "look",
    "watch",
    "point",
}

# Prepositions that usually encode spatial relations. We intentionally do not
# include broad prepositions such as "in", "with", or "of" here, because they
# caused many false positives: "man in red", "bowl of soup", etc.
SPATIAL_ADPOSITIONS = {
    "on",
    "under",
    "below",
    "above",
    "beneath",
    "over",
    "between",
    "behind",
    "beside",
    "near",
    "next",
    "around",
}

# Nouns that usually make "with X" an attribute/clothing description, not a
# true relation. Example: "guy with black shirt and dark jeans".
ATTIRE_OR_BODY_NOUNS = {
    "shirt",
    "tshirt",
    "t-shirt",
    "tee",
    "jeans",
    "pants",
    "shorts",
    "dress",
    "jacket",
    "coat",
    "sweater",
    "hoodie",
    "hat",
    "cap",
    "helmet",
    "glasses",
    "sunglasses",
    "hair",
    "beard",
    "mustache",
    "moustache",
    "face",
    "head",
    "back",
    "arm",
    "leg",
    "hand",
    "shoe",
    "shoes",
}

# Nouns that usually make "with X" a real object association.
ASSOCIATED_OBJECT_NOUNS = {
    "bag",
    "backpack",
    "purse",
    "laptop",
    "umbrella",
    "phone",
    "cellphone",
    "camera",
    "ball",
    "bat",
    "racket",
    "stick",
    "bottle",
    "cup",
    "plate",
    "kite",
}


class InputFileError(Exception):
    """Raised when an expected local input file or model is missing."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config YAML.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(VALID_SPLITS),
        help="RefCOCO-family dataset to classify.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "testA", "testB"],
        help="Dataset split to classify.",
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
            "Prepare or place the file there and rerun."
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


def normalize_expression(expression: str) -> str:
    normalized = expression.casefold()
    normalized = re.sub(r"[^a-z0-9+\- ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_expression(expression: str) -> list[str]:
    return normalize_expression(expression).split()


def contains_phrase(normalized: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None


def match_with_object_relation(expression: str) -> str | None:
    """Handle useful 'with X' cases without over-labeling clothing attributes."""
    tokens = tokenize_expression(expression)
    if "with" not in tokens:
        return None

    with_index = tokens.index("with")
    after_with = tokens[with_index + 1 :]
    if not after_with:
        return None

    if any(noun in after_with for noun in ASSOCIATED_OBJECT_NOUNS):
        return "with_object"

    if any(noun in after_with for noun in ATTIRE_OR_BODY_NOUNS):
        return None

    return None


def match_in_object_relation(expression: str) -> str | None:
    """Handle useful 'in X' object cases without over-labeling clothing attributes."""
    tokens = tokenize_expression(expression)
    if "in" not in tokens:
        return None

    in_index = tokens.index("in")
    after_in = tokens[in_index + 1 :]
    if not after_in:
        return None

    if any(noun in after_in for noun in ASSOCIATED_OBJECT_NOUNS):
        return "in_object"

    if any(noun in after_in for noun in ATTIRE_OR_BODY_NOUNS):
        return None

    return None


def match_relational_cue(expression: str) -> str | None:
    normalized = normalize_expression(expression)

    for cue in sorted(RELATIONAL_PHRASE_CUES, key=len, reverse=True):
        if contains_phrase(normalized, cue):
            return cue

    with_object_cue = match_with_object_relation(expression)
    if with_object_cue is not None:
        return with_object_cue

    in_object_cue = match_in_object_relation(expression)
    if in_object_cue is not None:
        return in_object_cue

    return None


def match_positional_cue(expression: str) -> str | None:
    normalized = normalize_expression(expression)
    tokens = tokenize_expression(expression)

    grid_patterns = [
        r"\b(?:top|upper|bottom|lower|middle|center|centre)\s+(?:left|right|middle|center|centre)\b",
        r"\b(?:left|right)\s+(?:top|upper|bottom|lower|middle|center|centre)\b",
        r"\b(?:far|farthest|furthest)\s+(?:left|right|top|bottom)\b",
    ]
    for pattern in grid_patterns:
        match = re.search(pattern, normalized)
        if match is not None:
            return match.group(0)

    for token in tokens:
        if token in POSITION_WORDS or token in ORDINAL_WORDS:
            return token

    if (
        "background" in tokens
        or contains_phrase(normalized, "back of frame")
        or contains_phrase(normalized, "in back")
        or contains_phrase(normalized, "back row")
    ):
        return "back"

    return None


@lru_cache(maxsize=1)
def load_spacy_model():
    try:
        import spacy
    except ImportError as exc:
        raise InputFileError(
            "spaCy is not installed. Install spaCy and the 'en_core_web_sm' model to use "
            "dependency fallback classification."
        ) from exc

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise InputFileError(
            "spaCy model 'en_core_web_sm' is not installed locally. Install it before running "
            "dependency fallback classification."
        ) from exc


def spans_overlap(span_a, span_b) -> bool:
    return span_a.start < span_b.end and span_b.start < span_a.end


def select_root_chunk(doc):
    noun_chunks = list(doc.noun_chunks)
    if not noun_chunks:
        return None

    root = doc[:].root
    for chunk in noun_chunks:
        if chunk.start <= root.i < chunk.end:
            return chunk

    ranked = []
    for chunk in noun_chunks:
        distance = 0
        node = chunk.root
        while node != node.head:
            if node.head == root:
                ranked.append((distance, chunk))
                break
            node = node.head
            distance += 1

    if ranked:
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]
    return noun_chunks[0]


def dependency_path_tokens(token_a, token_b):
    ancestors_a = [token_a, *token_a.ancestors]
    ancestors_b = [token_b, *token_b.ancestors]
    positions_b = {token: index for index, token in enumerate(ancestors_b)}

    common = None
    index_a = 0
    index_b = 0
    for idx, token in enumerate(ancestors_a):
        if token in positions_b:
            common = token
            index_a = idx
            index_b = positions_b[token]
            break

    if common is None:
        return []

    upward = ancestors_a[: index_a + 1]
    downward = list(reversed(ancestors_b[:index_b]))
    return upward + downward


def token_text_or_lemma(token) -> set[str]:
    return {token.text.casefold(), token.lemma_.casefold()}


def is_position_or_ordinal_root(token) -> bool:
    return bool(token_text_or_lemma(token) & (POSITION_WORDS | ORDINAL_WORDS))


def is_relational_token(token, other_root=None) -> bool:
    forms = token_text_or_lemma(token)
    if token.pos_ in {"VERB", "AUX"} and forms & RELATIONAL_VERBS:
        return True
    if (token.dep_ == "prep" or token.pos_ == "ADP") and forms & SPATIAL_ADPOSITIONS:
        if "on" in forms:
            return other_root is not None and not bool(
                token_text_or_lemma(other_root) & ATTIRE_OR_BODY_NOUNS
            )
        return True
    return False


def has_relational_dependency(expression: str) -> bool:
    """Conservative spaCy fallback.

    The original fallback returned relational for almost any prepositional or
    verbal connection between noun chunks. That caught true relations, but it
    also mislabeled many attributes such as "man in red", "bowl of soup", and
    "guy with black shirt". This version only accepts known spatial adpositions
    or known interaction verbs.
    """
    nlp = load_spacy_model()
    doc = nlp(expression)
    noun_chunks = list(doc.noun_chunks)
    if len(noun_chunks) < 2:
        return False

    root_chunk = select_root_chunk(doc)
    if root_chunk is None:
        return False

    for other_chunk in noun_chunks:
        if other_chunk == root_chunk or spans_overlap(root_chunk, other_chunk):
            continue
        if is_position_or_ordinal_root(other_chunk.root):
            continue

        path = dependency_path_tokens(root_chunk.root, other_chunk.root)
        if not path:
            continue

        interior = path[1:-1]
        if any(is_relational_token(token, other_chunk.root) for token in interior):
            return True

        if is_position_or_ordinal_root(other_chunk.root):
            continue
        ancestor = other_chunk.root
        while ancestor != ancestor.head and ancestor != root_chunk.root:
            if is_relational_token(ancestor, other_chunk.root):
                return True
            ancestor = ancestor.head

    return False


def classify_expression(expression: str) -> tuple[str, str]:
    matched_cue = match_relational_cue(expression)
    if matched_cue is not None:
        return "relational", matched_cue
    if has_relational_dependency(expression):
        return "relational", "spacy_dependency_conservative"
    positional_cue = match_positional_cue(expression)
    if positional_cue is not None:
        return "positional", positional_cue
    return "attribute", ""


def load_vg_relations(path: Path) -> dict[int, list]:
    mapping = {}
    for row in load_jsonl(path):
        coco_id = safe_int(row.get("coco_id"))
        if coco_id is None:
            continue
        relations = row.get("relations")
        mapping[coco_id] = relations if isinstance(relations, list) else []
    return mapping


def classify_split(dataset: str, split: str, config_path_str: str) -> int:
    if split not in VALID_SPLITS[dataset]:
        LOGGER.warning(
            "Split '%s' is not available for dataset '%s'; skipping without writing outputs.",
            split,
            dataset,
        )
        return 0

    config = load_config_paths(config_path_str)
    processed_dir = require_config_path(config, "processed_dir")
    splits_dir = require_config_path(config, "splits_dir")

    expressions_path = ensure_file(
        processed_dir / f"{dataset}_{split}.jsonl",
        f"processed {dataset}/{split} JSONL",
    )
    vg_mapping_path = ensure_file(
        processed_dir / "vg_relations_by_coco_id.jsonl",
        "Visual Genome coco_id relation mapping",
    )

    rows = load_jsonl(expressions_path)
    vg_relations = load_vg_relations(vg_mapping_path)

    total_rows = len(rows)
    filtered_rows = []
    for row in rows:
        coco_id = safe_int(row.get("coco_id"))
        relations = vg_relations.get(coco_id, [])
        if relations:
            filtered_rows.append(row)

    coverage = (len(filtered_rows) / total_rows * 100.0) if total_rows else 0.0
    LOGGER.info(
        "Coverage for %s/%s after VG relation filtering: %.2f%% (%d / %d expressions kept).",
        dataset,
        split,
        coverage,
        len(filtered_rows),
        total_rows,
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    relational_path = splits_dir / f"{dataset}_{split}_relational.jsonl"
    positional_path = splits_dir / f"{dataset}_{split}_positional.jsonl"
    attribute_path = splits_dir / f"{dataset}_{split}_attribute.jsonl"
    audit_path = splits_dir / f"{dataset}_{split}_classification_log.csv"

    relational_count = 0
    positional_count = 0
    attribute_count = 0

    with (
        relational_path.open("w", encoding="utf-8") as relational_handle,
        positional_path.open("w", encoding="utf-8") as positional_handle,
        attribute_path.open("w", encoding="utf-8") as attribute_handle,
        audit_path.open("w", encoding="utf-8", newline="") as audit_handle,
    ):
        audit_writer = csv.DictWriter(
            audit_handle,
            fieldnames=["ref_id", "expression", "label", "matched_cue"],
        )
        audit_writer.writeheader()

        for row in filtered_rows:
            expression = row.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                LOGGER.warning(
                    "Skipping malformed expression row with ref_id=%s because 'expression' is missing.",
                    row.get("ref_id"),
                )
                continue

            label, matched_cue = classify_expression(expression)
            output_row = dict(row)
            output_row["label"] = label
            output_row["matched_cue"] = matched_cue

            audit_writer.writerow(
                {
                    "ref_id": row.get("ref_id"),
                    "expression": expression,
                    "label": label,
                    "matched_cue": matched_cue,
                }
            )

            if label == "relational":
                relational_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                relational_count += 1
            elif label == "positional":
                positional_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                positional_count += 1
            else:
                attribute_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                attribute_count += 1

    LOGGER.info(
        "Finished %s/%s classification: %d relational, %d positional, %d attribute. Outputs: %s, %s, %s, %s",
        dataset,
        split,
        relational_count,
        positional_count,
        attribute_count,
        relational_path,
        positional_path,
        attribute_path,
        audit_path,
    )
    return 0


def main() -> int:
    args = parse_args()
    return classify_split(args.dataset, args.split, args.config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
