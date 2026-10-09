"""读取检查点里的 Rust tokenizer，不依赖 Transformers 或 PyTorch。"""

import json
from pathlib import Path

from tokenizers import Tokenizer as Backend


class Tokenizer:
    """提供序列构造需要的特殊 token 编号。"""

    def __init__(self, path):
        path = Path(path)
        self.backend = Backend.from_file(str(path / "tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        config = json.loads((path / "tokenizer_config.json").read_text())
        for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
            value = config.get(name)
            if isinstance(value, dict):
                value = value.get("content")
            token_id = self.backend.token_to_id(value) if isinstance(value, str) else None
            if token_id is None:
                raise ValueError("Tokenizer is missing a valid %s" % name)
            setattr(self, name, value)
            setattr(self, name + "_id", token_id)

    def __call__(self, text, add_special_tokens=False):
        encoded = self.backend.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": encoded.ids}
