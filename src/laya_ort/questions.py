"""把调用方的题目收成内部表示。"""

import json
from typing import Dict

from laya_ort.common import QTYPES, render_options


def _require_label(question_id: str, label) -> None:
    if isinstance(label, bool) or not isinstance(label, (str, int)):
        raise ValueError(
            "question %r: option labels must be strings or integers" % question_id
        )
    if isinstance(label, str) and not label.strip():
        raise ValueError("question %r: option labels must not be blank" % question_id)


def _choice_criteria(question_id: str, criteria):
    if isinstance(criteria, list):
        for label in criteria:
            _require_label(question_id, label)
        if len(criteria) != len(set(criteria)):
            raise ValueError("question %r: choice labels must be unique" % question_id)
        if not criteria:
            raise ValueError("question %r: choice needs at least one option" % question_id)
        return {label: None for label in criteria}
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("question %r: choice criteria must be a list or a non-empty object" % question_id)
    for label in criteria:
        _require_label(question_id, label)
    return criteria


def _score_criteria(question_id: str, criteria):
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("question %r: score criteria must be a non-empty list" % question_id)
    if any(level is None for level in criteria):
        raise ValueError("question %r: score levels must not be null" % question_id)
    return criteria


def _noul_criteria(question_id: str, criteria):
    if criteria is None:
        return None
    if not isinstance(criteria, dict):
        raise ValueError("question %r: noul criteria must be an object" % question_id)
    normalized = {}
    for key, value in criteria.items():
        name = str(key).lower()
        if name not in {"false", "true"}:
            raise ValueError(
                "question %r: noul criteria only accepts false and true" % question_id
            )
        normalized[name] = value
    return normalized


def normalize_question(question_id: str, question: Dict) -> Dict:
    """校验一道题，并收成序列构造使用的内部字典。"""
    if not isinstance(question, dict):
        raise TypeError("question %r must be an object" % question_id)
    kind = question.get("type")
    if kind not in QTYPES:
        raise ValueError("question %r: type must be choice, score, or noul" % question_id)
    if "instructions" not in question or question["instructions"] is None:
        raise ValueError("question %r: instructions are required" % question_id)
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    if instructions == "":
        raise ValueError("question %r: instructions must not be empty" % question_id)
    if kind == "choice":
        criteria = _choice_criteria(question_id, question.get("criteria"))
    elif kind == "score":
        criteria = _score_criteria(question_id, question.get("criteria"))
    else:
        criteria = _noul_criteria(question_id, question.get("criteria"))
    internal = {"t": kind, "ins": instructions, "crit": criteria}
    if "labels" in question:
        if kind != "noul":
            raise ValueError("question %r: labels is only supported for noul" % question_id)
        internal["labels"] = question["labels"]
    option_count = len(render_options(internal))
    if "option_order" in question:
        order = [int(index) for index in question["option_order"]]
        if sorted(order) != list(range(option_count)):
            raise ValueError(
                "question %r: option_order must be a permutation of the options" % question_id
            )
        internal["option_order"] = order
    return internal
