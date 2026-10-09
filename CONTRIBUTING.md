# Contributing

Thanks for looking at laya-ort. The useful change is a small one that keeps Linux x86_64 inference accurate and easy to run.

## Set up

```bash
git clone git@github.com:Fruank4/laya-ort.git
cd laya-ort
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Python 3.10 or newer. `laya-ort check` should print `ok`.

Do not commit `models/`, `.venv/`, or exported `*.onnx` graphs. Those files are large and are produced locally by `laya-ort export`.

## Pull requests

- Describe why the change is needed.
- Add or update a test when the change affects routing, sequence building, calibration, or the ONNX session.
- Keep the English `README.md` and Chinese `README.zh-CN.md` in step when the public behavior changes.
- Do not present a probe-graph result as accuracy on the official Laya weights.

Upstream behavior lives in [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). A change here should say whether it follows that project or is specific to this ONNX runtime.

## Conduct

This project uses the [Contributor Covenant](CODE_OF_CONDUCT.md).
