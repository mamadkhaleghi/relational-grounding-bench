import pytest

from data.classify_expressions import match_relational_cue


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
