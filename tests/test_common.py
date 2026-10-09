import numpy as np

from laya_ort.common import (
    build_sequence,
    clamp_temperature,
    collate_rows,
    temp_bucket,
)
from laya_ort.questions import normalize_question


class FakeTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 4
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        words = text.split()
        return {"input_ids": [10 + (index % 5) for index, _ in enumerate(words)] or [10]}


def test_sequence_marks_each_option():
    question = normalize_question("department", {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": ["billing", "technical"],
    })
    ids, markers, stats, state_stats = build_sequence(
        FakeTokenizer(),
        "please refund",
        question,
        max_len=64,
        head_max_len=32,
    )
    assert ids[0] == 1
    assert ids[-1] == 2
    assert markers == [index for index, token in enumerate(ids) if token == 4]
    assert len(markers) == 2
    assert stats["options"] == 2
    assert state_stats["truncated"] is False


def test_empty_instructions_are_rejected():
    try:
        normalize_question("q", {"type": "noul", "instructions": ""})
    except ValueError as exc:
        assert "instructions" in str(exc)
    else:
        raise AssertionError("empty instructions should fail")


def test_temperature_clamp_stops_overconfident_buckets():
    assert clamp_temperature(0.1006) == 0.5
    assert clamp_temperature(9) == 5.0
    assert temp_bucket(0, 77) == "choice:11+"


def test_collate_pads_short_rows():
    batch = collate_rows(
        [
            {"ids": [1, 2], "markers": [1], "qtype": 0},
            {"ids": [1, 2, 3, 4], "markers": [1, 3], "qtype": 2},
        ],
        pad_id=0,
    )
    assert batch["input_ids"].shape == (2, 4)
    assert batch["marker_pos"].shape == (2, 2)
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]
    assert np.array_equal(batch["qtype"], np.array([0, 2]))
