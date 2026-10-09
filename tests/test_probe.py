import pytest

from laya_ort.agent import load
from laya_ort.cli import self_test
from laya_ort.probe import write_probe_checkpoint

pytest.importorskip("onnx")


def test_probe_selects_the_later_option():
    result = self_test()
    assert result["answers"]["department"]["choice"] == "sales"
    assert result["usage"]["output_tokens"] == 0
    probabilities = result["answers"]["department"]["probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert result["answers"]["department"]["action"]["act_probability"] == 0.5


def test_probe_scores_a_rubric_and_a_yes_no(tmp_path):
    graph = write_probe_checkpoint(tmp_path)
    agent = load(str(tmp_path), str(graph))
    result = agent.predict(
        "please refund the duplicate charge today",
        {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"],
            },
            "churn_risk": {
                "type": "noul",
                "instructions": "Does the user threaten to cancel?",
            },
        },
    )
    assert result["answers"]["urgency"]["score"] > 1
    assert result["answers"]["churn_risk"]["noul"] > 0.5
    assert result["usage"]["output_tokens"] == 0


def test_none_state_is_rejected(tmp_path):
    graph = write_probe_checkpoint(tmp_path)
    agent = load(str(tmp_path), str(graph))
    with pytest.raises(ValueError, match="state"):
        agent.predict(None, {"q": {"type": "noul", "instructions": "Is it true?"}})
