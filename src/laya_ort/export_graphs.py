"""把三个官方检查点导出成 ONNX。需要另行安装 laya 和 PyTorch。"""

import shutil
from pathlib import Path
from typing import Optional

CHECKPOINTS = (
    ("english", "convaiinnovations/laya", None),
    ("multilingual", "convaiinnovations/laya", "multilingual"),
    ("typed-decisions", "convaiinnovations/laya", "typed-decisions"),
)


def export_checkpoints(output_dir: str = "models", only: Optional[str] = None) -> None:
    """导出英语、多语言和 typed-decisions 三张图，并把配置和分词器放在旁边。"""
    import torch
    from laya.agent import Agent

    destination = Path(output_dir)
    selected = [item for item in CHECKPOINTS if only is None or item[0] == only]
    if not selected:
        raise ValueError("unknown checkpoint %r" % only)
    for name, repo, subfolder in selected:
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        graph = folder / "laya.onnx"
        _copy_runtime_files(repo, subfolder, folder)
        if graph.is_file():
            print("keeping existing %s" % graph)
            continue
        print("loading %s for export" % name)
        kwargs = {"compile": False, "device": "cpu"}
        if subfolder and _accepts_subfolder(Agent):
            kwargs["subfolder"] = subfolder
            source = repo
        elif subfolder:
            source = str(folder)
        else:
            source = repo
        agent = Agent(source, **kwargs)
        agent.model.eval()
        _export_agent(agent, graph, torch)
        del agent
        print("wrote %s" % graph)


def _accepts_subfolder(agent_cls) -> bool:
    import inspect

    return "subfolder" in inspect.signature(agent_cls.__init__).parameters


def _copy_runtime_files(repo: str, subfolder: Optional[str], folder: Path) -> None:
    from huggingface_hub import snapshot_download

    prefix = "%s/" % subfolder if subfolder else ""
    snapshot = Path(snapshot_download(
        repo,
        allow_patterns=[prefix + "rl_agent_config.json", prefix + "tokenizer/*"],
    ))
    source = snapshot / subfolder if subfolder else snapshot
    config = source / "rl_agent_config.json"
    if not config.is_file():
        raise FileNotFoundError("missing rl_agent_config.json in %s" % source)
    shutil.copy2(config, folder / "rl_agent_config.json")
    tokenizer = source / "tokenizer"
    target = folder / "tokenizer"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(tokenizer, target)


def _export_agent(agent, graph: Path, torch) -> None:
    batch, seq_len, num_markers = 2, 17, 3
    inputs = (
        torch.randint(0, 100, (batch, seq_len), dtype=torch.long),
        torch.ones((batch, seq_len), dtype=torch.long),
        torch.tensor([[1, 5, 9]] * batch, dtype=torch.long),
        torch.ones((batch, num_markers), dtype=torch.bool),
        torch.zeros(batch, dtype=torch.long),
    )
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }
    torch.onnx.export(
        agent.model,
        inputs,
        str(graph),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
