# laya-ort

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-blue.svg)](https://github.com/Fruank4/laya-ort)

在 Linux x86_64 上运行 [Laya](https://github.com/NandhaKishorM/laya) 的类型化决策。Python 负责把状态和题目拼成序列、分词、校准，计算交给 ONNX Runtime 的 CPU。Intel 和 AMD 走同一条路径。一次前向给出概率，不生成文字。

Laya 是非自回归的决策模型：调用方当场给出选项，模型为每个选项打分。本仓库是这套推理在 ONNX 上的运行时，设计上对齐 [laya-mlx](https://github.com/mizorewww/laya-mlx) 的分层，后端从 MLX 换成了 ONNX Runtime。训练、HTTP 和苹果芯片不在这一版里。

```text
state + questions
        │
        ▼
Router          看文字，在三张图里选一张
        │
        ▼
序列构造        每个选项一个 [MASK]
        │
        ▼
ONNX Runtime CPU
        │
        ▼
choice / score / noul
```

## 快速开始

下面这段在三张图已经导出到 `models/` 之后可以直接跑。导出见 [导出检查点](#导出检查点)。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/laya-ort route --state "发票被重复扣款，请退款。"
.venv/bin/laya-ort predict \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

英语走 `models/english`，中文走 `models/multilingual`。`predict` 打印每个问题的选项、概率和 token 用量。输出 token 数是 0。

同一道退款题上，三张图选出的部门都是 `billing`，概率与上游 PyTorch 一致到小数点后四位。这是这条题目上的核对，不是完整评测。

## 三种题

一次 `predict` 可以同时回答多道题。标签写在请求里，换一批选项不需要重新训练。

| 类型 | 返回 | 适合 |
|---|---|---|
| `choice` | 一个选项和全部分布 | 分诊、路由、该交给哪个工具 |
| `score` | 有序量尺上的期望档位 | 紧急程度、严重度 |
| `noul` | 命题为真的概率 | 是否要求退款、是否越界 |

`examples/questions.json` 把这三种放在一起：

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

选项宜在 20 个以内，区分词放在开头。所有选项和说明共享一段 token 预算，选项太多时每个标签会被截短。

## 安装

Python 3.10 或更新版本。推理依赖是 `onnxruntime`、`tokenizers`、`numpy`、`huggingface_hub`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
laya-ort check
```

`check` 不下载权重。它用一张探针图走完拼题、会话和结果解码。探针把选项的位置当成分数，所以三选一总会选中最后一个选项 `sales`。成功时打印 `ok`。在 Linux x86_64 上还会打印 `target: linux x86_64`。

`[dev]` 额外安装 `pytest` 和 `onnx`。`onnx` 是探针图和导出要用的，预测已经导出的 `laya.onnx` 不需要它。

## 检查点

权重来自 Hugging Face 上的 `convaiinnovations/laya`。Git 仓库里没有这三张图。

| 名字 | 编码器 | 规模 | 上下文 | 图 |
|---|---|---:|---:|---|
| `english` | ModernBERT-large | 421M | 512 | `models/english/laya.onnx` |
| `multilingual` | mmBERT-base | 322M | 1024 | `models/multilingual/laya.onnx` |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | `models/typed-decisions/laya.onnx` |

导出后每个目录里还有 `rl_agent_config.json` 和 `tokenizer/`。预测读本地这些文件，不再下载 `model.safetensors`。三张 FP32 图大约是 1.6 GB、1.2 GB、1.6 GB。

`typed-decisions` 是在四组固定工作流上微调过的检查点。自动路由不会选它，除非传入 `--task typed_decisions`，或在 `Router(auto_task_detection=True)` 下题目 id 与某个工作流完全一致。

## 路由

`Router` 先决定用哪张图，再调用对应的 `Agent`。检测是纯 Python，不加载模型。

优先级：

1. 显式 `--model`，别名 `en`、`ml`、`typed` 都可以
2. 显式 `--task typed_decisions`
3. 打开自动工作流识别，且题目 id 与某个工作流完全一致
4. 显式 `--lang`，例如 `en`、`de`、`zh`
5. 文种和语言检测
6. 默认 `english`

非拉丁文字，包括中文，走多语言图。拉丁字母里看起来不是英语的文本也走多语言图。没有字母时用默认检查点。

```bash
laya-ort route --state "I was charged twice"
laya-ort route --state "发票被重复扣款，请退款。"
laya-ort route --state "I was charged twice" --lang de
laya-ort predict --state "I was charged twice" --questions examples/questions.json --task typed_decisions
```

`route` 只打印决定，不读 ONNX。同时驻留的图默认一张，避免把三份权重要进内存。

## 一次前向在做什么

每道题单独变成一条序列，状态会在每道题里各出现一次：

```text
[CLS] <题型> question: <说明> [SEP]
[MASK] 选项0 [MASK] 选项1 ... [SEP]
<状态> [SEP]
```

ONNX 图读五个输入：`input_ids`、`attention_mask`、`marker_pos`、`marker_mask`、`qtype`。输出是每个选项的 `logits`，以及行动头的 `act_logits`。运行时按题型和选项个数做温度缩放，再 softmax。温度钳在 0.5 到 5.0，避免上游 `choice:11+` 那种过锐的校准把接近随机的分布报成几乎确定。

会话固定使用 `CPUExecutionProvider`。

## Python

路由：

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

`result["routing"]` 里有 `model`、`reason`、`onnx`。`result["usage"]["output_tokens"]` 为 0。

已经知道要用哪张图时，可以跳过路由：

```python
from laya_ort import load

agent = load("models/english", "models/english/laya.onnx")
result = agent.predict("I was billed twice.", questions)
```

`questions` 的形状与上面的路由示例相同。多个状态用 `agent.predict_batch`。

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

`--onnx` 和 `--model` 一起用时不做路由，直接跑指定的那张图。`--model` 此时是检查点目录或 Hub id。

## 导出检查点

导出在装有 PyTorch 和上游 `laya` 的环境里做一次。Linux x86_64 上的预测不需要 PyTorch。

```bash
pip install laya onnx
laya-ort export --output models
```

只重导一张时：

```bash
laya-ort export --only multilingual
```

已经存在的 `laya.onnx` 会保留。配置和分词器会写到同一目录。Hub 上的大权重只在导出时下载。

## 开发

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

`tests/test_common.py` 覆盖序列、校准和题目校验。`tests/test_router.py` 覆盖选图规则，不加载 ONNX。`tests/test_probe.py` 用探针图跑通 ONNX Runtime。

## 这一版没有的东西

邮件清理、长文档扫描、HTTP、MCP、预测钩子、RLCD 微调和完整评测集还没有。置信度是校准后的最大概率，不是“这题一定对”。

## 许可

Apache-2.0。序列构造、温度分桶和语言检测改编自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) 与 [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)，详见 [NOTICE](NOTICE)。权重属于 Convai Innovations，不包含在本仓库中。
