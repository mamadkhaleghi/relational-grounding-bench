"""Shared VLM loading, prompt construction, and inference helpers."""

from __future__ import annotations

from typing import Any

from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import resize_long_edge


def load_vlm(config: dict):
    """Load the configured Qwen image-text-to-text model and processor."""
    model_config = config.get("model", {})
    base_model = model_config.get("base_model")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("Missing config entry 'model.base_model'.")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if model_config.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(base_model, **model_kwargs)
    processor = AutoProcessor.from_pretrained(base_model)
    max_long_edge = model_config.get("max_image_long_edge")
    if isinstance(max_long_edge, int) and max_long_edge > 0:
        processor.max_image_long_edge = max_long_edge
    return model, processor


def _relation_value(relation: dict, key: str) -> str:
    value = relation.get(key)
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())
    return "unknown"


def build_prompt(expression: str, relations: list[dict] | None) -> str:
    """Build a referring-expression prompt, optionally with known relations."""
    lines: list[str] = []
    if relations:
        lines.append("Known relations in this image:")
        for relation in relations:
            subject = _relation_value(relation, "subject")
            predicate = _relation_value(relation, "predicate")
            obj = _relation_value(relation, "object")
            lines.append(f"{subject} — {predicate} — {obj}")
        lines.append("")

    lines.extend(
        [
            "Locate the object described by this referring expression:",
            f'"{expression}"',
            "Return only one bounding box in this exact format: "
            "<box>(x1,y1),(x2,y2)</box>.",
            "Use absolute image pixel coordinates.",
        ]
    )
    return "\n".join(lines)


def _model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def run_inference(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Run one image-conditioned generation and return the decoded text."""
    import torch

    image = Image.open(Path(image_path)).convert("RGB")
    max_long_edge = getattr(processor, "max_image_long_edge", None)
    if isinstance(max_long_edge, int) and max_long_edge > 0:
        image = resize_long_edge(image, max_long_edge)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except ImportError:
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

    device = _model_device(model)
    if device is not None:
        inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_ids = inputs["input_ids"]
    generated_ids = [
        output_ids[len(input_ids[index]) :]
        for index, output_ids in enumerate(generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""
