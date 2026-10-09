<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <img src="docs/logo-light.svg" width="480" alt="laya-ort">
  </picture>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-2563EB" alt="Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+"></a>
  <a href="https://github.com/Fruank4/laya-ort"><img src="https://img.shields.io/badge/platform-Linux%20x86__64-0F172A" alt="Linux x86_64"></a>
  <a href="https://huggingface.co/convaiinnovations/laya"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20checkpoint-convaiinnovations%2Flaya-yellow" alt="上游检查点"></a>
</p>

**面向 Linux x86_64 的类型化决策。** [Laya](https://github.com/NandhaKishorM/laya) 在一次前向里给调用方当场给出的选项打分，不生成文字。本仓库用 ONNX Runtime 的 CPU 跑这个决策，Intel 和 AMD 走同一条路径。Python 负责拼题、分词和校准。

分层对齐 [laya-mlx](https://github.com/mizorewww/laya-mlx)：一层小的 Python 运行时，后面接一个本地计算后端。这里的后端是 ONNX，不是 MLX。训练、HTTP 和苹果芯片不在这一版里。

```text
状态 + 题目 → Router → 每题一条序列 → ONNX Runtime CPU → 概率
```

## 安装

```bash
python -m pip install "git+https://github.com/Fruank4/laya-ort.git"
```

使用 [uv](https://docs.astral.sh/uv/) 时，在 uv 项目里运行 `uv add "git+https://github.com/Fruank4/laya-ort.git"`，或在虚拟环境里运行 `uv pip install "git+https://github.com/Fruank4/laya-ort.git"`。

需要 Python 3.10 或更新版本。推理依赖是 ONNX Runtime、Hugging Face Tokenizers、NumPy 和 `huggingface_hub`。虚拟环境可以不碰系统 Python：

```bash
python -m venv .venv
.venv/bin/python -m pip install "git+https://github.com/Fruank4/laya-ort.git"
.venv/bin/laya-ort check
```

`check` 不下载权重。它用一张探针图走完拼题、ONNX 会话和结果解码。探针把选项的位置当成分数，所以三选一总会选中最后一个选项 `sales`。成功时打印 `ok`。在 Linux x86_64 上还会打印 `target: linux x86_64`。探针需要 `onnx` 包：`python -m pip install "laya-ort[dev] @ git+https://github.com/Fruank4/laya-ort.git"`。

## 快速开始

先把三个官方检查点导出一次，再提问。导出见 [下文](#导出检查点)。图很大，不放进 Git。

```bash
laya-ort route --state "发票被重复扣款，请退款。"
laya-ort predict \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

英语走 `models/english`，中文走 `models/multilingual`。`predict` 打印选中的标签、概率分布和 token 用量。输出 token 数是 0。

同一道退款题上，三张图都选出 `billing`，概率与上游 PyTorch 一致到小数点后四位。这是这一道题的核对，不是完整评测。

## 三种题

一次调用可以同时回答多道题。标签写在请求里，换一个选项不需要重新训练。

| 类型 | 返回 | 适合 |
|---|---|---|
| `choice` | 一个标签和全部分布 | 分诊、路由、该调用哪个工具 |
| `score` | 有序量尺上的期望档位 | 紧急程度、严重度 |
| `noul` | 命题为真的概率 | 是否要求退款、是否越界 |

`examples/questions.json` 把三种题放在一起：

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

`choice` 的 `criteria` 可以是上面这种带说明的对象，也可以是 `["billing", "technical", "sales"]`。`noul` 固定两个槽，先 `false` 后 `true`，返回值是 P(true)。

选项宜在 20 个以内，区分词放在开头。所有选项共享一段 token 预算，列表太长时标签会被截短，直到难以区分。

## 检查点

权重是 Convai Innovations 发布的 [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)。本 Git 仓库不包含这些文件。

| 名字 | 编码器 | 参数量 | 上下文 | 图 |
|---|---|---:|---:|---|
| `english` | ModernBERT-large | 421M | 512 | `models/english/laya.onnx` |
| `multilingual` | mmBERT-base | 322M | 1024 | `models/multilingual/laya.onnx` |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | `models/typed-decisions/laya.onnx` |

导出后，每个目录里还有 `rl_agent_config.json` 和 `tokenizer/`。预测读这些本地文件，不再下载 `model.safetensors`。三张 FP32 图大约是 1.6 GB、1.2 GB、1.6 GB。

`typed-decisions` 在四组固定工作流上微调过。自动路由不会选它，除非传入 `--task typed_decisions`，或使用 `Router(auto_task_detection=True)` 且题目 id 与某个工作流完全一致。

## 路由

`Router` 先选图，再调用那张图的 `Agent`。检测是纯 Python，不加载模型。

优先级：

1. 显式 `--model`。别名 `en`、`ml`、`typed` 都可以。
2. 显式 `--task typed_decisions`。
3. 打开工作流自动识别，且题目 id 与某个工作流完全一致。
4. 显式 `--lang`，例如 `en`、`de`、`zh`。
5. 文种和语言检测。
6. 默认 `english`。

非拉丁文字，包括中文，走多语言图。看起来不是英语的拉丁文本也走多语言图。没有字母时使用默认检查点。

```bash
laya-ort route --state "I was charged twice"
laya-ort route --state "发票被重复扣款，请退款。"
laya-ort route --state "I was charged twice" --lang de
laya-ort predict --state "I was charged twice" --questions examples/questions.json --task typed_decisions
```

`route` 只打印决定，不打开 ONNX 文件。默认同时只驻留一张图，避免三份权重一起占着内存。

## 一次前向

每道题各自变成一条序列。状态会在每道题里各出现一次：

```text
[CLS] <题型> question: <说明> [SEP]
[MASK] 选项0 [MASK] 选项1 ... [SEP]
<状态> [SEP]
```

图有五个输入：`input_ids`、`attention_mask`、`marker_pos`、`marker_mask`、`qtype`。输出是每个选项的 logit，以及行动头的 `act_logits`。运行时按题型和选项个数做温度缩放，再 softmax。温度钳在 0.5 到 5.0，所以上游 `choice:11+` 这种桶不能把接近平局的分布报成几乎确定。

会话只用 `CPUExecutionProvider`。

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

`result["routing"]` 里有 `model`、`reason` 和 `onnx`。`result["usage"]["output_tokens"]` 为 0。

已经知道要用哪张图时，可以跳过路由：

```python
from laya_ort import load

agent = load("models/english", "models/english/laya.onnx")
result = agent.predict(state, questions)
```

`questions` 的形状与上面的示例相同。多个状态用 `agent.predict_batch`。

## 命令行

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

`--onnx` 和 `--model` 一起用时不做路由，直接跑指定的那张图。这时 `--model` 是检查点目录或 Hub id。

## 导出检查点

导出在装有 PyTorch 和上游 `laya` 的环境里做一次。Linux x86_64 上的预测不需要 PyTorch。

```bash
python -m pip install laya onnx
laya-ort export --output models
```

只重导一张：

```bash
laya-ort export --only multilingual
```

已经存在的 `laya.onnx` 会保留。配置和分词器写到同一目录。Hub 上的大权重只在导出时下载。

## 开发

```bash
git clone git@github.com:Fruank4/laya-ort.git
cd laya-ort
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

`tests/test_common.py` 覆盖序列构造、校准和题目校验。`tests/test_router.py` 覆盖选图，不加载 ONNX。`tests/test_probe.py` 用探针图跑通 ONNX Runtime。

提交说明见 [CONTRIBUTING.md](CONTRIBUTING.md)，漏洞报告见 [SECURITY.md](SECURITY.md)。

## 这一版没有的东西

邮件清理、长文档扫描、HTTP、MCP、预测钩子、RLCD 微调和完整评测集还没有。置信度是校准后的最大概率，并不表示这题一定正确。

## 许可

[Apache-2.0](LICENSE)。序列构造、温度分桶和语言检测改编自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) 与 [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)，详见 [NOTICE](NOTICE)。权重属于 Convai Innovations，不在本仓库中。
