import pytest

from data.classify_expressions import classify_expression, match_relational_cue


@pytest.mark.parametrize(
    ("expression", "expected_label", "expected_cue"),
    [
        ("the dog left of the chair", "relational", "left of"),
        ("person right of the bicycle", "relational", "right of"),
        ("woman next to the table", "relational", "next to"),
        ("boy in front of the car", "relational", "in front of"),
        ("man holding a surfboard", "relational", "holding"),
        ("child riding the horse", "relational", "riding"),
        ("lamp above the desk", "relational", "above"),
        ("the red shirt", "attribute", None),
        ("small striped umbrella", "attribute", None),
        ("person with a blue backpack", "relational", "with_object"),
        ("man with a striped shirt", "attribute", None),
    ],
)
def test_match_relational_cue(expression, expected_label, expected_cue):
    matched_cue = match_relational_cue(expression)
    label = "relational" if matched_cue else "attribute"

    assert label == expected_label
    assert matched_cue == expected_cue


@pytest.mark.parametrize(
    ("expression", "expected_label"),
    [
        ("the dog left of the chair", "relational"),
        ("man holding a surfboard", "relational"),
        ("top carrot in bag", "relational"),
        ("blue jacket on woman", "attribute"),
        ("right bear", "positional"),
        ("left man", "positional"),
        ("top sandwich", "positional"),
        ("elephant on the right", "positional"),
        ("second plane", "positional"),
        ("the red shirt", "attribute"),
        ("small striped umbrella", "attribute"),
    ],
)
def test_classify_expression_three_way(expression, expected_label):
    assert classify_expression(expression)[0] == expected_label
