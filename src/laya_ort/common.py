"""题目序列、温度校准和批量对齐。序列格式与上游 Laya 一致。"""

import json
import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {value: name for name, value in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}

# 低于 0.5 的温度会把不确定的分布锐化成近乎确定。上游 choice:11+ 桶是 0.1006。
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def serialize_state(state: Union[str, dict, list]) -> str:
    """字符串原样保留，字典和对话列表收成 JSON。"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """把一个选项说明收成模型看见的文本。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def resolve_noul_labels(labels=None):
    """noul 只接受 false / true 两个显示词。"""
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    return false_label, true_label


def render_options(question: Dict) -> List[str]:
    """按选项下标渲染文本。noul 的顺序永远是 false 然后 true。"""
    kind, criteria = question["t"], question.get("crit")
    if kind != "noul" and "labels" in question:
        raise ValueError("labels is only supported for noul questions")
    if kind == "choice":
        return [
            str(key) if value is None or value == "" else "%s: %s" % (key, render_criterion(value))
            for key, value in criteria.items()
        ]
    if kind == "score":
        return ["level %d: %s" % (index, render_criterion(item)) for index, item in enumerate(criteria)]
    criteria = criteria or {}
    false_label, true_label = resolve_noul_labels(question.get("labels"))
    false_text, true_text = criteria.get("false"), criteria.get("true")
    return [
        false_label
        + ": "
        + (
            render_criterion(false_text)
            if false_text not in (None, "")
            else "no, the statement does not hold"
        ),
        true_label
        + ": "
        + (
            render_criterion(true_text)
            if true_text not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def build_prefix(tokenizer, question: Dict, head_max_len: int = 192, option_order=None):
    """构造状态之前的题面，并返回每个选项标记的位置。"""
    mask_token = tokenizer.mask_token
    options = render_options(question)
    order = option_order if option_order is not None else list(range(len(options)))
    instructions = str(question["ins"]).replace(mask_token, " ")
    head_ids = tokenizer(
        "%s question: %s" % (question["t"], instructions),
        add_special_tokens=False,
    )["input_ids"]
    option_ids = []
    for index in order:
        # 每个选项先占一个 [MASK]，正文最多再留 48 个 token。
        option_ids.append(
            [tokenizer.mask_token_id]
            + tokenizer(
                " " + options[index].replace(mask_token, " "),
                add_special_tokens=False,
            )["input_ids"][:48]
        )
    option_budget = head_max_len - sum(len(ids) for ids in option_ids)
    tokens_per_option = None
    if option_budget < 16:
        tokens_per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [ids[:tokens_per_option] for ids in option_ids]
        option_budget = head_max_len - sum(len(ids) for ids in option_ids)
    head_ids = head_ids[: max(8, option_budget)]
    ids = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    stats = {
        "options": len(option_ids),
        "options_distinct": len({tuple(ids) for ids in option_ids}),
        "tokens_per_option": tokens_per_option,
    }
    return ids, markers, stats


def finish_sequence(tokenizer, prefix, markers, state_ids, max_len, truncate_left=False):
    """把状态接到题面后面。对话列表保留末尾，其余文本保留开头。"""
    room = max(0, max_len - len(prefix) - 1)
    if truncate_left:
        kept = state_ids[max(0, len(state_ids) - room) :]
    else:
        kept = state_ids[:room]
    ids = (list(prefix) + kept + [tokenizer.sep_token_id])[:max_len]
    state_stats = {
        "state_tokens": len(state_ids),
        "state_tokens_used": len(kept),
        "state_tokens_dropped": len(state_ids) - len(kept),
        "truncated": len(kept) < len(state_ids),
    }
    return ids, [marker for marker in markers if marker < max_len], state_stats


def build_sequence(
    tokenizer,
    state,
    question,
    max_len=512,
    head_max_len=192,
    option_order=None,
    truncate_left=False,
    state_ids=None,
):
    """拼出一条决策序列，并带回选项统计和截断统计。"""
    prefix, markers, stats = build_prefix(tokenizer, question, head_max_len, option_order)
    if state_ids is None:
        text = serialize_state(state).replace(tokenizer.mask_token, " ")
        state_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids, markers, state_stats = finish_sequence(
        tokenizer, prefix, markers, state_ids, max_len, truncate_left
    )
    return ids, markers, stats, state_stats


def collapsed_options(question_ids, items) -> Dict[str, Dict[str, Optional[int]]]:
    """报告哪些题的选项被截成了相同的 token。"""
    report = {}
    for question_id, item in zip(question_ids, items):
        stats = item.get("options")
        if stats and stats["options_distinct"] < stats["options"]:
            report[question_id] = {
                "total": stats["options"],
                "distinct": stats["options_distinct"],
                "tokens_per_option": stats["tokens_per_option"],
            }
    return report


def temp_bucket(qtype: int, option_count: int) -> str:
    """温度按题型和选项个数分桶，分桶方式和上游相同。"""
    if option_count <= 2:
        size = "2"
    elif option_count <= 5:
        size = "3-5"
    elif option_count <= 10:
        size = "6-10"
    else:
        size = "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def clamp_temperature(value, low: float = TEMP_MIN, high: float = TEMP_MAX) -> float:
    """把温度限制在可用区间；非法值退回 1。"""
    if isinstance(value, bool):
        return 1.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return min(high, max(low, value))


def softmax(logits: np.ndarray) -> np.ndarray:
    """对一维 logits 做数值稳定的 softmax。"""
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def unpermute_probs(probabilities: np.ndarray, option_order: Optional[Sequence[int]]) -> np.ndarray:
    """把槽位顺序的概率搬回选项原来的下标。"""
    if option_order is None:
        return probabilities
    restored = np.empty(len(option_order), dtype=np.float64)
    for slot, original in enumerate(option_order):
        restored[original] = probabilities[slot]
    return restored


def answer_confidence(probabilities: np.ndarray, option_count: int) -> float:
    """校准后的最大选项概率。"""
    if option_count < 1:
        return 1.0
    return float(np.clip(np.max(probabilities[:option_count]), 0.0, 1.0))


def confidence_from_probs(probabilities: np.ndarray, option_count: int) -> float:
    """1 减去归一化熵。选项少于 2 个时没有不确定性。"""
    if option_count < 2:
        return 1.0
    kept = probabilities[:option_count]
    entropy = -(kept * np.log(np.clip(kept, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(option_count), 0.0, 1.0))


def collate_rows(rows: List[dict], pad_id: int) -> Dict[str, np.ndarray]:
    """把多道题垫成 ONNX 图要求的五个输入。"""
    width = max(len(row["ids"]) for row in rows)
    marker_width = max(len(row["markers"]) for row in rows)
    count = len(rows)
    input_ids = np.full((count, width), pad_id, dtype=np.int64)
    attention_mask = np.zeros((count, width), dtype=np.int64)
    marker_pos = np.zeros((count, marker_width), dtype=np.int64)
    marker_mask = np.zeros((count, marker_width), dtype=bool)
    qtype = np.zeros((count,), dtype=np.int64)
    for index, row in enumerate(rows):
        length = len(row["ids"])
        input_ids[index, :length] = row["ids"]
        attention_mask[index, :length] = 1
        marker_count = len(row["markers"])
        marker_pos[index, :marker_count] = row["markers"]
        marker_mask[index, :marker_count] = True
        qtype[index] = row["qtype"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }
