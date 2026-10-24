"""本地 BGE v1.5 编码：CLS pooling + L2 归一化；只有查询添加检索指令。"""
from __future__ import annotations

from pathlib import Path
import time

from langchain_core.embeddings import Embeddings


class LocalBGEEmbeddings(Embeddings):
    def __init__(self, model_name, *, revision, cache_dir, device=None,
                 batch_size=8, max_length=512,
                 query_instruction="为这个句子生成表示以用于检索相关文章："):
        if batch_size < 1 or not 2 <= max_length <= 512:
            raise ValueError("BGE batch_size 必须为正，max_length 必须在 2～512 之间")
        self.model_name, self.revision = model_name, revision
        self.cache_dir, self.device = Path(cache_dir), device
        self.batch_size, self.max_length = batch_size, max_length
        self.query_instruction = query_instruction
        self._model = self._tokenizer = None
        self.load_seconds = 0.0

    def warmup(self):
        if self._model is not None:
            return self.load_seconds
        started = time.perf_counter()
        print(f"正在加载本地 Embedding：{self.model_name} …", flush=True)
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        path = Path(self.model_name)
        if not path.is_dir():
            options = dict(repo_id=self.model_name, revision=self.revision,
                           cache_dir=str(self.cache_dir), token=False,
                           allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin",
                                           "tokenizer.json", "tokenizer_config.json",
                                           "special_tokens_map.json", "vocab.txt"])
            try:
                path = Path(snapshot_download(**options, local_files_only=True))
                # snapshot 目录也可能来自上次未完成的下载；仅有目录不代表权重齐全。
                required = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
                if (not all((path / name).is_file() for name in required)
                        or not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))):
                    raise LocalEntryNotFoundError("BGE 本地快照不完整")
            except LocalEntryNotFoundError:
                print("首次运行：从 Hugging Face 下载 BGE 权重到项目 model_cache/，只需一次。", flush=True)
                path = Path(snapshot_download(**options, max_workers=2))
        # 使用本地 snapshot 路径，不在每次加载时联网检查模型版本。
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = AutoModel.from_pretrained(str(path), local_files_only=True, dtype=torch.float32)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = tokenizer
        self._model = model.to(self.device).eval()
        self._encode(["预热"])
        self.load_seconds = time.perf_counter() - started
        print(f"Embedding 就绪：{self.device}，{model.config.hidden_size} 维，"
              f"加载及预热 {self.load_seconds:.1f} 秒（本地推理，无 API 费用）", flush=True)
        return self.load_seconds

    def _encode(self, texts):
        if not texts:
            return []
        self.warmup()
        import torch
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self._tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                vectors = self._model(**inputs).last_hidden_state[:, 0].float()
                if not torch.isfinite(vectors).all() or torch.any(vectors.norm(dim=1) == 0):
                    raise ValueError("BGE 返回了无效向量")
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            result.extend(vectors.cpu().tolist())
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 文档与语义窗口均不添加 query instruction。
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_instruction + text])[0]
