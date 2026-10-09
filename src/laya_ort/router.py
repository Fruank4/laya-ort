"""按文字在三张 ONNX 图之间选择检查点。检测逻辑来自 laya-mlx。"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from laya_ort.lang import analyse

BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}
CHECKPOINT_NAMES = ("english", "multilingual", "typed-decisions")

_ALIASES = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "decisions": "typed-decisions",
}

_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {
        "discrepancy_severity",
        "disposition",
        "duplicate",
        "matches_order",
        "urgency",
    },
    "security_incidents": {
        "credential_compromise",
        "disposition",
        "severity",
        "true_positive",
        "urgency",
    },
}

_ENGLISH_SUBTAGS = ("en", "eng", "english")
_LANGUAGE_AGNOSTIC_CODES = ("c", "posix", "und", "zxx", "mul")


def _repo_str(spec) -> str:
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


def _split(spec):
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None


class RouteDecision(dict):
    """路由结果：用哪张图，以及为什么。"""

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    """把别名收成 english、multilingual 或 typed-decisions。"""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES))
        )
    return key


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """题目 id 与某个固定工作流完全一致时，返回工作流名字。"""
    ids = set(questions or {})
    for name, signature in _TYPED_DECISION_WORKFLOWS.items():
        if ids == signature:
            return name
    return None


def _english_from_code(value: Any) -> Optional[bool]:
    """语言代码是否英语。空代码和 C/POSIX 返回 None，交给检测。"""
    if value is None:
        return None
    code = str(value).strip().lower()
    if not code:
        return None
    code = code.split(".", 1)[0]
    primary = code.replace("_", "-").split("-", 1)[0]
    if not primary or primary in _LANGUAGE_AGNOSTIC_CODES:
        return None
    return primary in _ENGLISH_SUBTAGS


class Router:
    """在三张已导出的 ONNX 图之间按文字选一张，再做预测。"""

    def __init__(
        self,
        onnx_dir: Union[str, Path] = "models",
        onnx: Optional[Dict[str, str]] = None,
        models: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        max_loaded: int = 1,
        default: str = "english",
        auto_task_detection: bool = False,
        preload: bool = False,
    ):
        self.onnx_dir = Path(onnx_dir)
        self._onnx = {normalise_name(key): Path(path) for key, path in (onnx or {}).items()}
        self.models = dict(DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(key): value for key, value in models.items()})
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        if preload:
            self.preload()

    def graph_path(self, name: str) -> Path:
        """这张检查点对应的 ONNX 文件。"""
        key = normalise_name(name)
        if key in self._onnx:
            return self._onnx[key]
        return self.onnx_dir / key / "laya.onnx"

    def checkpoint_dir(self, name: str):
        """导出目录里已有配置时用本地文件，否则回退到 Hub 子目录。"""
        key = normalise_name(name)
        local = self.onnx_dir / key
        if (local / "rl_agent_config.json").is_file() and (local / "tokenizer").is_dir():
            return str(local), None
        return _split(self.models[key])

    def load(self, name: str):
        """加载这张图。同时驻留的数量受 max_loaded 限制。"""
        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            from laya_ort.agent import Agent

            graph = self.graph_path(key)
            if not graph.is_file():
                raise FileNotFoundError(
                    "ONNX graph for %s is not at %s. Run `laya-ort export` first."
                    % (key, graph)
                )
            repo, subfolder = self.checkpoint_dir(key)
            agent = Agent(repo, str(graph), subfolder=subfolder, token=self.token)
            self._agents[key] = agent
            self._order.append(key)
            self._evict()
            return agent

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict(self):
        with self._lock:
            while len(self._order) > self.max_loaded:
                victim = self._order.pop(0)
                self._agents.pop(victim, None)

    def preload(self, names: Optional[List[str]] = None):
        """提前加载，避免第一次请求才读图。"""
        selected = [normalise_name(name) for name in (list(self.models) if names is None else names)]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(set(selected) | set(self._agents)))
            for name in selected:
                if name not in self._agents:
                    self.load(name)
        return self

    def unload(self, name: Optional[str] = None):
        """卸掉一张图，或全部卸掉。"""
        with self._lock:
            if name is None:
                self._agents.clear()
                self._order.clear()
                return
            key = normalise_name(name)
            self._agents.pop(key, None)
            if key in self._order:
                self._order.remove(key)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> RouteDecision:
        """决定用哪张图。这一步不加载模型。"""
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                onnx=str(self.graph_path(key)),
                reason="explicit model=%r" % model,
                detection=None,
                workflow=None,
            )
        if task is not None:
            key = normalise_name(
                "typed-decisions"
                if str(task).lower().replace("-", "_") == "typed_decisions"
                else task
            )
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                onnx=str(self.graph_path(key)),
                reason="explicit task=%r" % task,
                detection=None,
                workflow=None,
            )
        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(
                model="typed-decisions",
                repo=_repo_str(self.models["typed-decisions"]),
                onnx=str(self.graph_path("typed-decisions")),
                reason="question ids match the %r typed-decisions workflow" % workflow,
                detection=None,
                workflow=workflow,
            )
        resolved = _english_from_code(lang)
        if resolved is not None:
            key = "english" if resolved else "multilingual"
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                onnx=str(self.graph_path(key)),
                reason="explicit lang=%r" % lang,
                detection=None,
                workflow=workflow,
            )
        detected = analyse(state)
        if detected["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif detected["script"] != "latin":
            key = "multilingual"
            reason = (
                "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                % (detected["script"], 100 * float(detected["non_latin_fraction"]))
            )
        elif not detected["is_english"]:
            key = "multilingual"
            if detected["language"]:
                reason = "Latin script but language looks like %r, not English" % detected["language"]
            else:
                reason = (
                    "Latin script, language not identified but %.0f%% non-English letters; "
                    "not safe for the English checkpoint" % (100 * float(detected["diacritic_rate"]))
                )
        elif detected["language_undecided"]:
            key = self.default
            reason = "Latin language undecided; using default (%s)" % key
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(
            model=key,
            repo=_repo_str(self.models[key]),
            onnx=str(self.graph_path(key)),
            reason=reason,
            detection=detected,
            workflow=workflow,
        )

    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """先选图，再在选中的检查点上回答。"""
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        agent = self.load(decision["model"])
        result = agent.predict(state, questions)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (
            self.loaded,
            self.max_loaded,
            self.default,
        )
