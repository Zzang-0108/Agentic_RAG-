"""用重排模型 tokenizer 计量长度；不将字符数冒充 token 数。"""
from __future__ import annotations
import hashlib
from functools import lru_cache
from pathlib import Path


class TokenCounter:
    def __init__(self, tokenizer, model_name):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.signature = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf8")).hexdigest()

    @classmethod
    def load(cls, model_name, revision=None):
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        path = Path(model_name)
        if not path.is_dir():
            try:
                # 使用本地 snapshot 路径，避免库为已缓存的 tokenizer 再请求远程配置。
                path = Path(snapshot_download(model_name, revision=revision, local_files_only=True))
            except LocalEntryNotFoundError:
                raise RuntimeError(f"缺少切分 tokenizer 缓存：{model_name}。请先下载该 tokenizer，或设置 RAG_CHUNK_TOKENIZER 为本地目录。") from None
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        return cls(tokenizer, model_name)

    @lru_cache(maxsize=8192)
    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))


def bounded_end(text, start, end, count, max_tokens, max_chars=None):
    """以字符位置裁剪，保留原文，不通过 decode 重构导致来源 offset 漂移。"""
    high = min(end, start + max_chars) if max_chars else end
    if count(text[start:high]) <= max_tokens:
        return high
    low = start
    while low + 1 < high:
        middle = (low + high) // 2
        if count(text[start:middle]) <= max_tokens:
            low = middle
        else:
            high = middle
    if low == start:
        raise ValueError("长度预算不足以容纳单个字符")
    return low

