"""构造一张探针图，用来确认 CPU 执行链路，不代表真实 Laya 权重。"""

import json
from pathlib import Path


def build_probe(path) -> None:
    """logits 等于选项标记的位置，所以靠后的选项得分更高。"""
    import onnx
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info("marker_pos", TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, [None, None]),
        helper.make_tensor_value_info("qtype", TensorProto.INT64, [None]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None]),
        helper.make_tensor_value_info("act_logits", TensorProto.FLOAT, [None, 2]),
    ]
    two = helper.make_tensor("two", TensorProto.INT64, [1], [2])
    nodes = [
        helper.make_node("Cast", ["marker_pos"], ["logits"], to=TensorProto.FLOAT),
        helper.make_node("Shape", ["qtype"], ["batch_shape"]),
        helper.make_node("Concat", ["batch_shape", "two"], ["act_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["act_shape"],
            ["act_logits"],
            value=helper.make_tensor("fill", TensorProto.FLOAT, [1], [0.0]),
        ),
    ]
    graph = helper.make_graph(nodes, "laya_probe", inputs, outputs, initializer=[two])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def write_word_tokenizer(directory: Path) -> None:
    """写一个只认特殊 token 的分词器，探针测试不需要真实词表。"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    specials = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    vocab = {token: index for index, token in enumerate(specials)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    directory.mkdir(parents=True, exist_ok=True)
    backend.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
    }))


def write_probe_checkpoint(directory: Path) -> Path:
    """在临时目录放上配置、分词器和探针图。"""
    directory.mkdir(parents=True, exist_ok=True)
    write_word_tokenizer(directory / "tokenizer")
    (directory / "rl_agent_config.json").write_text(json.dumps({
        "max_len": 128,
        "head_max_len": 64,
        "temperature": [1.0, 1.0, 1.0],
        "option_layout": "sequential",
    }))
    graph = directory / "probe.onnx"
    build_probe(graph)
    return graph
