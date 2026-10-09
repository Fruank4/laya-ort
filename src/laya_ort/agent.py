"""用 ONNX Runtime 的 CPU 执行一次类型化决策。"""

import json
import os
import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from laya_ort.common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    answer_confidence,
    clamp_temperature,
    collapsed_options,
    collate_rows,
    confidence_from_probs,
    serialize_state,
    softmax,
    temp_bucket,
    unpermute_probs,
    build_sequence,
    render_criterion,
    render_options,
)
from laya_ort.questions import normalize_question
from laya_ort.tokenizer import Tokenizer


def resolve_checkpoint(model: str, subfolder: Optional[str] = None, token: Optional[str] = None) -> Path:
    """本地目录直接使用；Hub 仓库只下载配置和 tokenizer，不下载权重。"""
    path = Path(model)
    if path.exists():
        if subfolder:
            path = path / subfolder
        if not path.is_dir():
            raise FileNotFoundError("Checkpoint directory not found: %s" % path)
        return path
    if model.startswith(("/", "./", "../")) or os.path.isabs(model):
        raise FileNotFoundError("Local model path not found: %r" % model)
    from huggingface_hub import snapshot_download

    prefix = "%s/" % subfolder if subfolder else ""
    downloaded = snapshot_download(
        model,
        token=token or os.environ.get("HF_TOKEN") or None,
        allow_patterns=[
            prefix + "rl_agent_config.json",
            prefix + "tokenizer/*",
        ],
    )
    path = Path(downloaded)
    if subfolder:
        path = path / subfolder
    if not path.is_dir():
        raise FileNotFoundError("Subfolder %r not found in %r" % (subfolder, model))
    return path


class Agent:
    """一次前向读完状态和全部选项，返回 choice、score 或 noul。"""

    def __init__(
        self,
        model: str,
        onnx_path: str,
        subfolder: Optional[str] = None,
        token: Optional[str] = None,
    ):
        model_dir = resolve_checkpoint(model, subfolder, token=token)
        config_path = model_dir / "rl_agent_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                "Checkpoint %s has no rl_agent_config.json" % model_dir
            )
        self.cfg = json.loads(config_path.read_text())
        if self.cfg.get("option_layout", "sequential") != "sequential":
            raise ValueError(
                "option_layout=%r is not supported" % self.cfg.get("option_layout")
            )
        graph = Path(onnx_path)
        if not graph.is_file():
            raise FileNotFoundError(
                "ONNX model not found at %r. Export it with upstream scripts/export_onnx.py first."
                % onnx_path
            )
        self.tok = Tokenizer(model_dir / "tokenizer")
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        if len(self.temperature_raw) != 3:
            raise ValueError("Checkpoint temperature must contain 3 values")
        self.temperature = [clamp_temperature(value) for value in self.temperature_raw]
        self.temperature_by_options = {
            key: clamp_temperature(value)
            for key, value in self.temperature_by_options_raw.items()
        }
        self._warn_clamped_temperatures()
        self._session = _open_session(graph)
        self._lock = threading.Lock()

    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, dict],
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> Dict:
        """对一个状态回答一组题目。"""
        return self.predict_batch(
            [state],
            questions,
            max_len=max_len,
            head_max_len=head_max_len,
        )[0]

    system_one = predict

    def predict_batch(
        self,
        states: List[Union[str, dict, list]],
        questions: Dict[str, dict],
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        batch_size: int = 16,
    ) -> List[Dict]:
        """多个状态共用同一次会话，按 batch_size 分块。"""
        if not isinstance(questions, dict):
            raise TypeError("questions must be an object")
        for state in states:
            if state is None:
                raise ValueError("state must not be None")
            if not isinstance(state, (str, dict, list)):
                raise TypeError("state must be a string, object, or list")
        question_ids = list(questions)
        internal = {
            question_id: normalize_question(question_id, questions[question_id])
            for question_id in question_ids
        }
        max_len, head_max_len = self._budgets(max_len, head_max_len)
        if not question_ids:
            return [_empty_result() for _ in states]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results = []
        for start in range(0, len(states), batch_size):
            chunk = states[start : start + batch_size]
            encoded = [
                self._encode_state(state, question_ids, internal, max_len, head_max_len)
                for state in chunk
            ]
            results.extend(self._run_chunk(encoded, question_ids, internal))
        return results

    def _warn_clamped_temperatures(self) -> None:
        rejected = [
            "%s=%.4g" % (key, float(value))
            for key, value in self.temperature_by_options_raw.items()
            if clamp_temperature(value) != float(value)
        ]
        rejected += [
            "temperature[%d]=%.4g" % (index, float(value))
            for index, value in enumerate(self.temperature_raw)
            if clamp_temperature(value) != float(value)
        ]
        if rejected:
            warnings.warn(
                "laya-ort: temperatures outside [%g, %g] are clamped: %s"
                % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning,
                stacklevel=2,
            )

    def _budgets(self, max_len, head_max_len):
        max_len = self.cfg.get("max_len", 512) if max_len is None else int(max_len)
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else int(head_max_len)
        if not 4 < head_max_len < max_len:
            raise ValueError("Expected 4 < head_max_len < max_len")
        return max_len, head_max_len

    def _encode_state(self, state, question_ids, internal, max_len, head_max_len):
        truncate_left = isinstance(state, list)
        text = serialize_state(state).replace(self.tok.mask_token, " ")
        state_ids = self.tok(text, add_special_tokens=False)["input_ids"]
        items = []
        for question_id in question_ids:
            question = internal[question_id]
            sequence, markers, stats, state_stats = build_sequence(
                self.tok,
                state,
                question,
                max_len,
                head_max_len,
                option_order=question.get("option_order"),
                truncate_left=truncate_left,
                state_ids=state_ids,
            )
            option_count = len(render_options(question))
            if len(markers) != option_count:
                raise ValueError(
                    "question %r: only %d of its %d option markers fit in max_len=%d "
                    "with head_max_len=%d spent on the question"
                    % (question_id, len(markers), option_count, max_len, head_max_len)
                )
            items.append({
                "ids": sequence,
                "markers": markers,
                "qtype": QTYPES[question["t"]],
                "options": stats,
                "state_stats": state_stats,
            })
        return items

    def _run_chunk(self, encoded, question_ids, internal):
        rows = [item for items in encoded for item in items]
        batch = collate_rows(rows, self.tok.pad_token_id)
        _pad_markers(batch)
        with self._lock:
            logits, act_logits = self._session.run(
                ["logits", "act_logits"],
                batch,
            )
        action = softmax_rows(np.asarray(act_logits, dtype=np.float32))
        logits = np.asarray(logits, dtype=np.float32)
        results = []
        row = 0
        for items in encoded:
            count = len(items)
            dropped = max(item["state_stats"]["state_tokens_dropped"] for item in items)
            usage = {
                "input_tokens": int(batch["attention_mask"][row : row + count].sum()),
                "output_tokens": 0,
                "state_tokens": items[0]["state_stats"]["state_tokens"],
                "state_tokens_dropped": dropped,
                "truncated": dropped > 0,
                "truncated_questions": [
                    question_id
                    for question_id, item in zip(question_ids, items)
                    if item["state_stats"]["truncated"]
                ],
            }
            collapsed = collapsed_options(question_ids, items)
            if collapsed:
                usage["options"] = collapsed
            results.append({
                "model": "laya-ort",
                "answers": self._decode(logits, action, items, question_ids, internal, row),
                "usage": usage,
            })
            row += count
        return results

    def _decode(self, logits, action, items, question_ids, internal, offset):
        answers = {}
        for index, question_id in enumerate(question_ids):
            question = internal[question_id]
            option_count = len(items[index]["markers"])
            qtype = QTYPES[question["t"]]
            bucket = temp_bucket(qtype, option_count)
            temperature = self.temperature_by_options.get(bucket, self.temperature[qtype])
            probabilities = softmax(logits[offset + index, :option_count] / temperature)
            probabilities = unpermute_probs(probabilities, question.get("option_order"))
            confidence = round(confidence_from_probs(probabilities, option_count), 4)
            selected = round(answer_confidence(probabilities, option_count), 4)
            extra = {"act_probability": round(float(action[offset + index, 0]), 4)}
            if question["t"] == "choice":
                keys = list(question["crit"].keys())
                answers[question_id] = {
                    "type": "choice",
                    "choice": keys[int(np.argmax(probabilities))],
                    "probabilities": {
                        key: round(float(value), 4)
                        for key, value in zip(keys, probabilities)
                    },
                    "confidence": confidence,
                    "answer_confidence": selected,
                    "action": extra,
                }
            elif question["t"] == "score":
                expected = float((np.arange(option_count) * probabilities).sum())
                answers[question_id] = {
                    "type": "score",
                    "score": round(expected, 4),
                    "legend": {
                        str(level): render_criterion(text)
                        for level, text in enumerate(question["crit"])
                    },
                    "probabilities": {
                        str(level): round(float(value), 4)
                        for level, value in enumerate(probabilities)
                    },
                    "confidence": confidence,
                    "answer_confidence": selected,
                    "action": extra,
                }
            else:
                yes = float(probabilities[1])
                answers[question_id] = {
                    "type": "noul",
                    "noul": round(yes, 4),
                    "confidence": round(max(yes, 1.0 - yes), 4),
                    "answer_confidence": selected,
                    "action": extra,
                }
        return answers


def load(
    model: str,
    onnx_path: str,
    subfolder: Optional[str] = None,
    token: Optional[str] = None,
) -> Agent:
    """加载检查点配置、tokenizer 和 ONNX 图。"""
    return Agent(model, onnx_path, subfolder=subfolder, token=token)


def _open_session(graph: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    available = set(ort.get_available_providers())
    if "CPUExecutionProvider" not in available:
        raise RuntimeError("ONNX Runtime has no CPUExecutionProvider")
    return ort.InferenceSession(
        str(graph),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _pad_markers(batch: Dict[str, np.ndarray]) -> None:
    """导出的图对选项概率取前两名，单选项题要补一个无效槽。"""
    if batch["marker_pos"].shape[1] >= 2:
        return
    width = ((0, 0), (0, 2 - batch["marker_pos"].shape[1]))
    batch["marker_pos"] = np.pad(batch["marker_pos"], width)
    batch["marker_mask"] = np.pad(batch["marker_mask"], width)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """对每一行做 softmax。行动头至少有一列。"""
    if logits.ndim == 1:
        logits = logits[:, None]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def _empty_result() -> Dict:
    return {
        "model": "laya-ort",
        "answers": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
