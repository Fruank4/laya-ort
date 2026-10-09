from laya_ort.router import Router


QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": ["billing", "technical", "sales"],
    }
}


def test_english_text_uses_the_english_graph():
    decision = Router().route("I was charged twice and the payment is not refunded.", QUESTIONS)
    assert decision["model"] == "english"
    assert decision["onnx"].endswith("models/english/laya.onnx")


def test_chinese_text_uses_the_multilingual_graph():
    decision = Router().route("发票被重复扣款，请退款。", QUESTIONS)
    assert decision["model"] == "multilingual"
    assert decision["reason"].startswith("non-Latin script")


def test_explicit_language_and_model_win():
    router = Router()
    assert router.route("发票被重复扣款", QUESTIONS, lang="en")["model"] == "english"
    assert router.route("I was charged twice", QUESTIONS, lang="de")["model"] == "multilingual"
    assert router.route("发票被重复扣款", QUESTIONS, model="ml")["model"] == "multilingual"


def test_typed_decisions_workflow_is_opt_in():
    questions = {
        "action": {},
        "category": {},
        "churn_risk": {},
        "needs_human": {},
        "urgency": {},
    }
    assert Router().route("I was charged twice", questions)["model"] == "english"
    routed = Router(auto_task_detection=True).route("I was charged twice", questions)
    assert routed["model"] == "typed-decisions"
    assert routed["workflow"] == "customer_service"


def test_missing_graph_is_reported_before_inference(tmp_path):
    router = Router(onnx_dir=tmp_path)
    try:
        router.load("english")
    except FileNotFoundError as exc:
        assert "laya-ort export" in str(exc)
    else:
        raise AssertionError("missing ONNX graph should fail")
