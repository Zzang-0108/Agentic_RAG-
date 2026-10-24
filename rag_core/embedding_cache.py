"""按模型标识和完整文本缓存向量；每批提交，API 中断后可复用已完成批次。"""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np


class CachedEmbeddings:
    def __init__(self, backend, path, model_name, batch_size=8):
        self.backend, self.path, self.model_name = backend, Path(path), model_name
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("embedding batch_size 必须为正")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")

    def key(self, text):
        return hashlib.sha256((self.model_name + "\0" + text).encode("utf8")).hexdigest()

    def embed_documents(self, texts):
        keys = [self.key(t) for t in texts]
        values, missing = {}, {}
        with sqlite3.connect(self.path) as db:
            for key, text in zip(keys, texts):
                row = db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
                if row is not None:
                    values[key] = json.loads(row[0])
                else:
                    missing[key] = text
        pending = list(missing.items())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            vectors = self.backend.embed_documents([t for _, t in batch])
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(batch) or not np.isfinite(matrix).all():
                raise ValueError("Embedding 返回数量/维度错误或非有限值")
            if np.any(np.linalg.norm(matrix, axis=1) == 0):
                raise ValueError("Embedding 返回零向量")
            with sqlite3.connect(self.path) as db:
                for (key, _), vector in zip(batch, matrix.tolist()):
                    db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (key, json.dumps(vector)))
                    values[key] = vector
        result = [values[key] for key in keys]
        if result:
            matrix = np.asarray(result, dtype=np.float32)
            if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
                raise ValueError("缓存向量无效或模型输出维度发生变化，请使用新的模型缓存标识")
        return result

    def embed_query(self, text):
        return self.backend.embed_query(text)
