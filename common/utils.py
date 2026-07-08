"""Common utility helpers for configuration, logging, geometry, and parsing."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_QWEN_BOX_PATTERN = re.compile(
    rf"<box>\s*\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s*,\s*"
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s*</box>",
    re.IGNORECASE,
)
_JSON_BBOX_PATTERN = re.compile(
    rf'"bbox"\s*:\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]',
    re.IGNORECASE,
)


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load a YAML config file into a dictionary."""
    config_path = Path(path)
    if not config_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / config_path
        if candidate.exists():
            config_path = candidate

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return data if data is not None else {}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name: str) -> logging.Logger:
    """Create or reuse an INFO-level stream logger with timestamps."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)

    return logger


def compute_iou(box_a, box_b) -> float:
    """Compute IoU for [x1, y1, x2, y2] absolute-pixel boxes."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


def parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract a bounding box from VLM free text output."""
    match = _QWEN_BOX_PATTERN.search(text)
    if match:
        return tuple(float(value) for value in match.groups())

    match = _JSON_BBOX_PATTERN.search(text)
    if match:
        return tuple(float(value) for value in match.groups())

    return None


def resize_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    """Resize an image only when its longer edge exceeds the given limit."""
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image

    scale = max_long_edge / long_edge
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize(new_size, resample)
