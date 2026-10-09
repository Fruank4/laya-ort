<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <img src="docs/logo-light.svg" width="480" alt="laya-ort">
  </picture>
</p>

<p align="center">
  <a href="README.zh-CN.md"><strong>中文</strong></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-2563EB" alt="Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+"></a>
  <a href="https://github.com/Fruank4/laya-ort"><img src="https://img.shields.io/badge/platform-Linux%20x86__64-0F172A" alt="Linux x86_64"></a>
  <a href="https://huggingface.co/convaiinnovations/laya"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20checkpoint-convaiinnovations%2Flaya-yellow" alt="Upstream checkpoint"></a>
</p>

**Typed decisions for Linux x86_64.** [Laya](https://github.com/NandhaKishorM/laya) scores the options you supply in one forward pass and does not generate text. This repository runs that decision on ONNX Runtime's CPU, for Intel and AMD machines. Python builds the question, tokenizes it, and calibrates the probabilities.

The layout follows [laya-mlx](https://github.com/mizorewww/laya-mlx): a small Python runtime in front of a native compute backend. Here the backend is ONNX instead of MLX. Training, HTTP, and Apple Silicon are outside this release.

```text
state + questions → Router → one sequence per question → ONNX Runtime CPU → probabilities
```

## Installation

```bash
python -m pip install "git+https://github.com/Fruank4/laya-ort.git"
```

With [uv](https://docs.astral.sh/uv/), run `uv add "git+https://github.com/Fruank4/laya-ort.git"` in a uv project, or `uv pip install "git+https://github.com/Fruank4/laya-ort.git"` in a virtual environment.

Python 3.10 or newer. The runtime dependencies are ONNX Runtime, Hugging Face Tokenizers, NumPy, and `huggingface_hub`. A virtual environment keeps that install off the system Python:

```bash
python -m venv .venv
.venv/bin/python -m pip install "git+https://github.com/Fruank4/laya-ort.git"
.venv/bin/laya-ort check
```

`check` does not download weights. It runs one probe graph through question building, the ONNX session, and result decoding. The probe treats an option's position as its score, so a three-way choice selects the last label, `sales`. Success prints `ok`. On Linux x86_64 it also prints `target: linux x86_64`. The probe needs the `onnx` package: `python -m pip install "laya-ort[dev] @ git+https://github.com/Fruank4/laya-ort.git"`.

## Quickstart

Export the three official checkpoints once, then ask a question. Export is documented [below](#export-the-checkpoints). The graphs are large and are not stored in git.

```bash
laya-ort route --state "发票被重复扣款，请退款。"
laya-ort predict \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

English text uses `models/english`. Chinese uses `models/multilingual`. `predict` prints the chosen label, the distribution, and token usage. The output token count is 0.

On one refund question, all three graphs selected `billing`, and the probabilities matched upstream PyTorch to four decimal places. That check is for this question, not a full benchmark.

## Three question types

One call can answer several questions. The labels are part of the request, so a new option does not need a new training run.

| Type | Returns | Use it for |
|---|---|---|
| `choice` | one label and the full distribution | triage, routing, which tool to call |
| `score` | the expected level on an ordered rubric | urgency, severity |
| `noul` | the probability that a proposition is true | refund requested, policy violation |

`examples/questions.json` asks all three:

```json
{
  "department": {
    "type": "choice",
    "instructions": "Which department should handle this?",
    "criteria": {
      "billing": "invoices, payments, refunds",
      "technical": "bugs, outages, system errors",
      "other": "everything else"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["not urgent", "soon", "blocking"]
  },
  "churn_risk": {
    "type": "noul",
    "instructions": "Does the user threaten to cancel or leave?"
  }
}
```

A `choice` criterion may be that object, or a list such as `["billing", "technical", "sales"]`. A `noul` question has two slots, `false` then `true`, and the returned number is P(true).

Keep a choice under about 20 options, and put the distinguishing words first. Every option shares one token budget, so a long list is truncated until the labels are hard to tell apart.

## Checkpoints

The weights are Convai Innovations' release, [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) on Hugging Face. This git repository does not contain them.

| Name | Encoder | Parameters | Context | Graph |
|---|---|---:|---:|---|
| `english` | ModernBERT-large | 421M | 512 | `models/english/laya.onnx` |
| `multilingual` | mmBERT-base | 322M | 1024 | `models/multilingual/laya.onnx` |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | `models/typed-decisions/laya.onnx` |

After export, each directory also holds `rl_agent_config.json` and `tokenizer/`. Prediction reads those local files and does not download `model.safetensors`. The three FP32 graphs are about 1.6 GB, 1.2 GB, and 1.6 GB.

`typed-decisions` is fine-tuned on four fixed workflows. Routing leaves it alone unless you pass `--task typed_decisions`, or you construct `Router(auto_task_detection=True)` and the question ids match one workflow exactly.

## Routing

`Router` chooses a graph, then calls that graph's `Agent`. Detection is plain Python and does not load a model.

Precedence:

1. An explicit `--model`. Aliases `en`, `ml`, and `typed` are accepted.
2. An explicit `--task typed_decisions`.
3. Opt-in workflow detection, when the question ids match a workflow exactly.
4. An explicit `--lang`, such as `en`, `de`, or `zh`.
5. Script and language detection.
6. The default, `english`.

Non-Latin text, including Chinese, uses the multilingual graph. Latin text that does not look like English uses it too. Text with no letters keeps the default.

```bash
laya-ort route --state "I was charged twice"
laya-ort route --state "发票被重复扣款，请退款。"
laya-ort route --state "I was charged twice" --lang de
laya-ort predict --state "I was charged twice" --questions examples/questions.json --task typed_decisions
```

`route` prints the decision and does not open an ONNX file. One graph stays resident by default, so three weight sets are not held at once.

## One forward pass

Each question becomes its own sequence. The state is repeated in every question:

```text
[CLS] <type> question: <instructions> [SEP]
[MASK] option0 [MASK] option1 ... [SEP]
<state> [SEP]
```

The graph takes five inputs: `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, and `qtype`. It returns one logit per option and the action head's `act_logits`. The runtime divides by a temperature that depends on the question type and the option count, then applies softmax. Temperatures are clamped to the range 0.5 through 5.0, so an upstream bucket such as `choice:11+` cannot turn a near-tie into a reported certainty.

The session uses `CPUExecutionProvider` only.

## Python

```python
from laya_ort import Router

router = Router(onnx_dir="models")
result = router.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": ["billing", "technical", "sales"],
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
    },
)
print(result["routing"]["model"])
print(result["answers"]["department"]["choice"])
print(result["answers"]["urgency"]["score"])
print(result["answers"]["churn_risk"]["noul"])
```

`result["routing"]` carries `model`, `reason`, and `onnx`. `result["usage"]["output_tokens"]` is 0.

To skip routing and open one graph yourself:

```python
from laya_ort import load

agent = load("models/english", "models/english/laya.onnx")
result = agent.predict(state, questions)
```

`questions` has the same shape as the example above. Several states go through `agent.predict_batch`.

## Command line

```bash
laya-ort check
laya-ort route --state TEXT [--state-file FILE] [--questions FILE]
               [--model NAME] [--lang CODE] [--task typed_decisions]
laya-ort predict --questions FILE (--state TEXT | --state-file FILE)
                 [--onnx-dir models] [--model NAME] [--lang CODE] [--task NAME]
laya-ort predict --model PATH_OR_HUB --onnx FILE --questions FILE --state TEXT
                 [--subfolder NAME]
laya-ort export [--output models] [--only english|multilingual|typed-decisions]
```

Passing `--onnx` together with `--model` skips routing and runs that file. In that form, `--model` is a checkpoint directory or a Hub id.

## Export the checkpoints

Export runs where PyTorch and upstream `laya` are installed. Prediction on Linux x86_64 does not need PyTorch.

```bash
python -m pip install laya onnx
laya-ort export --output models
```

To export one checkpoint again:

```bash
laya-ort export --only multilingual
```

An existing `laya.onnx` is left in place. The config and tokenizer are written beside it. The large Hub weights are downloaded only for export.

## Development

```bash
git clone git@github.com:Fruank4/laya-ort.git
cd laya-ort
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

`tests/test_common.py` covers sequence building, calibration, and question checks. `tests/test_router.py` covers graph selection and does not load ONNX. `tests/test_probe.py` runs the probe graph through ONNX Runtime.

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull requests and [SECURITY.md](SECURITY.md) for vulnerability reports.

## Not in this release

Email cleaning, long-document scans, HTTP, MCP, prediction hooks, RLCD fine-tuning, and a full evaluation suite are not included. Confidence is a calibrated max probability. It does not mean the answer is certainly correct.

## License

[Apache-2.0](LICENSE). Sequence construction, temperature buckets, and language detection are adapted from [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) and [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx). See [NOTICE](NOTICE). The weights belong to Convai Innovations and are not in this repository.
