import csv

from PIL import Image
import pytest

from common.utils import compute_iou, parse_bbox_from_text, resize_long_edge
from finetune.train_qlora import (
    InputFileError,
    append_run_log,
    resolve_resume_checkpoint,
)

CURRENT_RUN_LOG_FIELDNAMES = [
    "rank",
    "freeze_vision_tower",
    "peak_mem_mb",
    "train_seconds",
    "output_dir",
    "resumed_from",
    "max_train_samples",
]


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args)


def test_compute_iou_identical_boxes():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_compute_iou_disjoint_boxes():
    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_compute_iou_partial_overlap():
    expected = 25.0 / 175.0
    assert compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) == expected


def test_parse_bbox_from_qwen_box_text():
    text = "<box>(329,270,420,635)"
    assert parse_bbox_from_text(text) == (329.0, 270.0, 420.0, 635.0)


def test_parse_bbox_from_paired_qwen_box_text():
    text = "<box>(10,20),(50,90)</box>"
    assert parse_bbox_from_text(text) == (10.0, 20.0, 50.0, 90.0)


def test_parse_bbox_from_json_text():
    text = 'Prediction: {"bbox": [1.5, 2, 3.25, 4]}'
    assert parse_bbox_from_text(text) == (1.5, 2.0, 3.25, 4.0)


def test_parse_bbox_from_unparseable_text():
    assert parse_bbox_from_text("No bounding box was provided.") is None


def test_resize_long_edge_resizes_preserving_aspect_ratio():
    image = Image.new("RGB", (400, 200), color="white")
    resized = resize_long_edge(image, 300)

    assert resized.size == (300, 150)


def test_resize_long_edge_leaves_smaller_image_unchanged():
    image = Image.new("RGB", (120, 80), color="white")
    resized = resize_long_edge(image, 300)

    assert resized.size == (120, 80)


def test_resume_from_checkpoint_omitted_starts_fresh(tmp_path):
    logger = RecordingLogger()

    resolved = resolve_resume_checkpoint(None, tmp_path, logger)

    assert resolved is None
    assert logger.messages == ["Checkpoint resume mode: fresh start"]


def test_resume_from_checkpoint_auto_uses_latest_checkpoint(tmp_path):
    (tmp_path / "checkpoint-5").mkdir()
    latest = tmp_path / "checkpoint-12"
    latest.mkdir()
    logger = RecordingLogger()

    resolved = resolve_resume_checkpoint("auto", tmp_path, logger)

    assert resolved == str(latest.resolve())
    assert logger.messages == [
        f"Checkpoint resume mode: auto-detected; resuming from {latest.resolve()} "
        "at step 12"
    ]


def test_resume_from_checkpoint_auto_starts_fresh_when_none_exists(tmp_path):
    logger = RecordingLogger()

    resolved = resolve_resume_checkpoint("auto", tmp_path, logger)

    assert resolved is None
    assert logger.messages == [
        f"Checkpoint resume mode: auto; no checkpoint found in {tmp_path}; "
        "starting fresh"
    ]


def test_resume_from_explicit_checkpoint_logs_saved_step(tmp_path):
    checkpoint = tmp_path / "custom-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 23}', encoding="utf-8"
    )
    logger = RecordingLogger()

    resolved = resolve_resume_checkpoint(str(checkpoint), tmp_path, logger)

    assert resolved == str(checkpoint.resolve())
    assert logger.messages == [
        f"Checkpoint resume mode: explicit; resuming from {checkpoint.resolve()} "
        "at step 23"
    ]


def test_resume_from_explicit_checkpoint_rejects_missing_directory(tmp_path):
    missing = tmp_path / "checkpoint-99"

    with pytest.raises(InputFileError, match="does not exist"):
        resolve_resume_checkpoint(str(missing), tmp_path, RecordingLogger())


def read_run_log(log_path):
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_resume_run_log_migrates_legacy_v1_rows(tmp_path):
    log_path = tmp_path / "finetune_run_log.csv"
    log_path.write_text(
        "rank,freeze_vision_tower,peak_mem_mb,train_seconds,output_dir\n"
        "4,True,100.00,10.00,checkpoints/legacy\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint-20"

    append_run_log(
        tmp_path,
        rank=8,
        freeze=True,
        peak_mem_mb=200.0,
        train_seconds=20.0,
        output_dir=tmp_path / "adapter",
        resumed_from=str(checkpoint),
    )

    fieldnames, rows = read_run_log(log_path)

    assert fieldnames == CURRENT_RUN_LOG_FIELDNAMES
    assert rows == [
        {
            "rank": "4",
            "freeze_vision_tower": "True",
            "peak_mem_mb": "100.00",
            "train_seconds": "10.00",
            "output_dir": "checkpoints/legacy",
            "resumed_from": "",
            "max_train_samples": "",
        },
        {
            "rank": "8",
            "freeze_vision_tower": "True",
            "peak_mem_mb": "200.00",
            "train_seconds": "20.00",
            "output_dir": str(tmp_path / "adapter"),
            "resumed_from": str(checkpoint),
            "max_train_samples": "",
        },
    ]


def test_resume_run_log_migrates_legacy_v2_rows(tmp_path):
    log_path = tmp_path / "finetune_run_log.csv"
    log_path.write_text(
        "rank,freeze_vision_tower,peak_mem_mb,train_seconds,output_dir,resumed_from\n"
        "4,False,100.00,10.00,checkpoints/legacy,checkpoint-10\n",
        encoding="utf-8",
    )

    append_run_log(
        tmp_path,
        rank=8,
        freeze=True,
        peak_mem_mb=200.0,
        train_seconds=20.0,
        output_dir=tmp_path / "adapter",
        resumed_from="checkpoint-20",
        max_train_samples=4000,
    )

    fieldnames, rows = read_run_log(log_path)

    assert fieldnames == CURRENT_RUN_LOG_FIELDNAMES
    assert rows == [
        {
            "rank": "4",
            "freeze_vision_tower": "False",
            "peak_mem_mb": "100.00",
            "train_seconds": "10.00",
            "output_dir": "checkpoints/legacy",
            "resumed_from": "checkpoint-10",
            "max_train_samples": "",
        },
        {
            "rank": "8",
            "freeze_vision_tower": "True",
            "peak_mem_mb": "200.00",
            "train_seconds": "20.00",
            "output_dir": str(tmp_path / "adapter"),
            "resumed_from": "checkpoint-20",
            "max_train_samples": "4000",
        },
    ]


def test_resume_run_log_appends_to_current_schema(tmp_path):
    log_path = tmp_path / "finetune_run_log.csv"
    log_path.write_text(
        ",".join(CURRENT_RUN_LOG_FIELDNAMES)
        + "\n4,False,100.00,10.00,checkpoints/current,checkpoint-10,2000\n",
        encoding="utf-8",
    )

    append_run_log(
        tmp_path,
        rank=8,
        freeze=True,
        peak_mem_mb=200.0,
        train_seconds=20.0,
        output_dir=tmp_path / "adapter",
        resumed_from=None,
        max_train_samples=None,
    )

    fieldnames, rows = read_run_log(log_path)

    assert fieldnames == CURRENT_RUN_LOG_FIELDNAMES
    assert rows == [
        {
            "rank": "4",
            "freeze_vision_tower": "False",
            "peak_mem_mb": "100.00",
            "train_seconds": "10.00",
            "output_dir": "checkpoints/current",
            "resumed_from": "checkpoint-10",
            "max_train_samples": "2000",
        },
        {
            "rank": "8",
            "freeze_vision_tower": "True",
            "peak_mem_mb": "200.00",
            "train_seconds": "20.00",
            "output_dir": str(tmp_path / "adapter"),
            "resumed_from": "",
            "max_train_samples": "",
        },
    ]


def test_resume_run_log_rejects_reordered_schema(tmp_path):
    log_path = tmp_path / "finetune_run_log.csv"
    reordered_fieldnames = CURRENT_RUN_LOG_FIELDNAMES.copy()
    reordered_fieldnames[0], reordered_fieldnames[1] = (
        reordered_fieldnames[1],
        reordered_fieldnames[0],
    )
    log_path.write_text(",".join(reordered_fieldnames) + "\n", encoding="utf-8")

    with pytest.raises(InputFileError, match="Unexpected columns"):
        append_run_log(
            tmp_path,
            rank=8,
            freeze=True,
            peak_mem_mb=200.0,
            train_seconds=20.0,
            output_dir=tmp_path / "adapter",
            resumed_from=None,
        )
