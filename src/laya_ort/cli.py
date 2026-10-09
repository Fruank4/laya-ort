"""命令行：检查 CPU 链路，或对一份 ONNX 检查点做预测。"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from laya_ort.platform import describe_host, is_linux_x86_64


def main(argv=None) -> int:
    """解析子命令。"""
    parser = argparse.ArgumentParser(prog="laya-ort")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="run one CPU forward pass on a probe graph")
    predict = sub.add_parser("predict", help="score a state, routing across the three ONNX graphs")
    predict.add_argument("--model", help="checkpoint name, directory, or Hub id")
    predict.add_argument("--onnx", help="one exported laya.onnx; skips routing")
    predict.add_argument("--onnx-dir", default="models", help="directory with english/, multilingual/, typed-decisions/")
    predict.add_argument("--state", help="state text")
    predict.add_argument("--state-file", help="UTF-8 file used as the state")
    predict.add_argument("--questions", required=True, help="JSON file of typed questions")
    predict.add_argument("--subfolder", help="checkpoint subfolder when --onnx is set")
    predict.add_argument("--lang", help="explicit language code, such as en or de")
    predict.add_argument("--task", help="typed_decisions selects that checkpoint")
    route = sub.add_parser("route", help="print which checkpoint would be used")
    route.add_argument("--state", help="state text")
    route.add_argument("--state-file", help="UTF-8 file used as the state")
    route.add_argument("--questions", help="optional JSON file of typed questions")
    route.add_argument("--model", help="force a checkpoint name")
    route.add_argument("--lang", help="explicit language code")
    route.add_argument("--task", help="typed_decisions selects that checkpoint")
    route.add_argument("--onnx-dir", default="models")
    export = sub.add_parser("export", help="export the three official checkpoints to ONNX")
    export.add_argument("--output", default="models")
    export.add_argument("--only", choices=["english", "multilingual", "typed-decisions"])
    args = parser.parse_args(argv)
    if args.command == "check":
        return check()
    if args.command == "route":
        return route_command(args)
    if args.command == "export":
        return export_command(args)
    return predict_command(args)


def check() -> int:
    """在当前进程里跑通一次探针决策。"""
    host = describe_host()
    if is_linux_x86_64():
        print("target: linux x86_64 (%s)" % host)
    else:
        print("host: %s" % host)
        print("supported deployment target: linux x86_64")
    try:
        result = self_test()
    except ModuleNotFoundError as exc:
        print("self-test needs the dev extra: pip install 'laya-ort[dev]' (%s)" % exc.name)
        return 1
    choice = result["answers"]["department"]["choice"]
    print("probe choice: %s" % choice)
    print("output_tokens: %s" % result["usage"]["output_tokens"])
    if choice != "sales" or result["usage"]["output_tokens"] != 0:
        print("probe forward did not match the expected decision")
        return 1
    print("ok")
    return 0


def self_test() -> dict:
    """用探针图回答一道三选一。靠后的选项位置更大，所以应选 sales。"""
    from laya_ort.agent import load
    from laya_ort.probe import write_probe_checkpoint

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        graph = write_probe_checkpoint(root)
        agent = load(str(root), str(graph))
        return agent.predict(
            "I was billed twice. Please refund the duplicate.",
            {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this?",
                    "criteria": ["billing", "technical", "sales"],
                }
            },
        )


def _state_from_args(args):
    if args.state is None and args.state_file is None:
        print("pass --state or --state-file", file=sys.stderr)
        return None
    if args.state is not None and args.state_file is not None:
        print("pass only one of --state and --state-file", file=sys.stderr)
        return None
    if args.state is not None:
        return args.state
    return Path(args.state_file).read_text()


def route_command(args) -> int:
    """只打印路由，不加载 ONNX。"""
    from laya_ort.router import Router

    state = _state_from_args(args)
    if state is None:
        return 2
    questions = {}
    if args.questions:
        questions = json.loads(Path(args.questions).read_text())
    decision = Router(onnx_dir=args.onnx_dir).route(
        state,
        questions,
        model=args.model,
        task=args.task,
        lang=args.lang,
    )
    json.dump(decision, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def export_command(args) -> int:
    """导出三张官方检查点。"""
    from laya_ort.export_graphs import export_checkpoints

    try:
        export_checkpoints(args.output, only=args.only)
    except ModuleNotFoundError as exc:
        print("export needs PyTorch and laya installed: pip install laya (%s)" % exc.name, file=sys.stderr)
        return 1
    return 0


def predict_command(args) -> int:
    """读题目文件，打印一次预测的 JSON。"""
    state = _state_from_args(args)
    if state is None:
        return 2
    questions = json.loads(Path(args.questions).read_text())
    if args.onnx:
        from laya_ort.agent import load

        if not args.model:
            print("pass --model with --onnx", file=sys.stderr)
            return 2
        result = load(args.model, args.onnx, subfolder=args.subfolder).predict(state, questions)
    else:
        from laya_ort.router import Router

        result = Router(onnx_dir=args.onnx_dir).predict(
            state,
            questions,
            model=args.model,
            task=args.task,
            lang=args.lang,
        )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0
