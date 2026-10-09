# laya-ort

在 **Linux x86_64** 上跑 Laya 的类型化决策。Python 负责拼题、分词和校准，计算交给 **ONNX Runtime 的 CPU**。Intel 和 AMD 都走这条路径。这一版不做训练，也不接 MLX、Core ML 或苹果芯片。

一次调用回答三类题，不生成文字：

- `choice`：在给定选项里选一个
- `score`：在有序量尺上给期望档位
- `noul`：命题为真的概率

```text
state + questions → 序列（每个选项一个 [MASK]）→ ONNX Runtime CPU → 概率
```

## 安装

需要 Python 3.10 或更新版本。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/laya-ort check
```

`check` 用一张探针图走完拼题、会话和结果解码。探针图把选项位置当成分数，所以三选一总会选中最后一个选项 `sales`。它证明 CPU 链路是通的，不是 Laya 权重的准确率。

在 Linux x86_64 上，成功时打印 `target: linux x86_64` 和 `ok`。

## 用真实权重预测

权重仍来自上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)。导出需要另装 PyTorch 和 `laya`，三张图写到 `models/`：

```bash
pip install laya
laya-ort export --output models
laya-ort route --state "发票被重复扣款，请退款。"
laya-ort predict \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

`predict` 不带 `--onnx` 时按文字选图。非拉丁文字走 `models/multilingual/laya.onnx`，英语走 `models/english/laya.onnx`。`typed-decisions` 只在 `--task typed_decisions` 时使用。

只跑一张已经导出的图时仍然可以指定文件：

```bash
laya-ort predict \
  --model convaiinnovations/laya \
  --onnx laya.onnx \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

Hub 上的 `model.safetensors` 不会在预测时下载。导出目录里已经有配置和分词器时，预测也不再访问 Hub。

Python：

```python
from laya_ort import load

agent = load("convaiinnovations/laya", "laya.onnx")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(result["answers"]["department"]["choice"])
```

会话固定使用 `CPUExecutionProvider`。输出 token 数是 0。

三张图在 `models/english`、`models/multilingual` 和 `models/typed-decisions`，体积大约 1.6 GB、1.2 GB、1.6 GB，不进 Git。同一道退款题上，英语图、多语言图和 typed-decisions 图选出的选项与上游 PyTorch 一致。

## 这一版没有的东西

邮件清理、长文档、HTTP、微调和整套评测还没有。温度仍按上游的分桶使用，并钳在 0.5 到 5.0。
