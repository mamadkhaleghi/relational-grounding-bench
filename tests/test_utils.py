from PIL import Image

from common.utils import compute_iou, parse_bbox_from_text, resize_long_edge


def test_compute_iou_identical_boxes():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_compute_iou_disjoint_boxes():
    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_compute_iou_partial_overlap():
    expected = 25.0 / 175.0
    assert compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) == expected


def test_parse_bbox_from_qwen_box_text():
    text = "The object is at <box>(12,34),(56,78)</box> in the image."
    assert parse_bbox_from_text(text) == (12.0, 34.0, 56.0, 78.0)


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
