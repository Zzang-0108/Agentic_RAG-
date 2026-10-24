"""corpus_hash 的黄金样本。

这个哈希是「向量缓存能不能复用」的唯一判据，写法必须和重构前逐字节一致
（json.dumps 的默认分隔符、list-of-3-lists 的结构、metadata 直接下标取值）。
这里锁死两个值：

1. 固定的两篇假文档 -> 固定的哈希（下面这个常量）
2. 真实语料的哈希 -> 由 rag.py 在启动时打印（见 README），改动语料就会变

万一哪天有人顺手把 json.dumps 加了 separators= 或 sort_keys=True，
这个测试会立刻炸，而不是等到某个用户白花一次 embedding 的钱。
"""
import os
import sys
import unittest
import json
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402

from rag_core.indexing import corpus_hash, make_cache_key  # noqa: E402
from rag_core import store
from rag_core.config import EMBEDDING_CONFIG

# 这两篇文档算出来的哈希（2026-10-18 由重构前的代码与重构后的代码分别算出，一致）
GOLDEN_HASH = "dd84d2f0d0534b9ab5743320f991f0fb2097a45237db9d9e5c0356c946e3c9e7"

# 真实语料（4 本 PDF、3924 个片段）的哈希，用来和旧评测报告对齐
REAL_CORPUS_HASH = "d64a01410360a3d4a98da31f68cca98bfab22fa7ad56637795adb9e2e7e4a1e5"


def golden_splits():
    return [
        Document(page_content="甲", metadata={"source": "knowledge_db/book/a.pdf", "page_number": 1}),
        Document(page_content="乙", metadata={"source": "knowledge_db/book/b.pdf", "page_number": 2}),
    ]


class TestCorpusHash(unittest.TestCase):
    def test_golden_value(self):
        self.assertEqual(corpus_hash(golden_splits()), GOLDEN_HASH)

    def test_order_matters(self):
        forward, backward = golden_splits()
        self.assertNotEqual(
            corpus_hash([forward, backward]), corpus_hash([backward, forward])
        )

    def test_content_and_page_and_source_all_count(self):
        base = golden_splits()
        for field, value in (("source", "knowledge_db/book/c.pdf"), ("page_number", 99)):
            metadata = dict(base[0].metadata, **{field: value})
            edited = [Document(page_content="甲", metadata=metadata), base[1]]
            self.assertNotEqual(corpus_hash(edited), GOLDEN_HASH, field)

        edited = [Document(page_content="丙", metadata=dict(base[0].metadata)), base[1]]
        self.assertNotEqual(corpus_hash(edited), GOLDEN_HASH)

    def test_extra_metadata_does_not_count(self):
        """score 之类的附加字段不参与哈希（不然查一次就变一次，缓存永远失效）。"""
        splits = golden_splits()
        splits[0].metadata["score"] = 0.87
        self.assertEqual(corpus_hash(splits), GOLDEN_HASH)

    def test_hash_stays_a_plain_hex_string(self):
        self.assertRegex(GOLDEN_HASH, r"^[0-9a-f]{64}$")


class TestCacheKey(unittest.TestCase):
    def test_model_encoding_config_is_recorded(self):
        """保留原字段，并记录本地模型版本及编码约定供缓存隔离和评测追溯。"""
        key = make_cache_key(GOLDEN_HASH, 2)
        self.assertEqual(
            sorted(key), ["chunk_count", "corpus_hash", "embed_model", "embedding_config", "schema_version", "split_config"]
        )
        self.assertEqual(key["chunk_count"], 2)
        self.assertEqual(key["corpus_hash"], GOLDEN_HASH)
        self.assertIsInstance(key["split_config"], dict)  # 必须是普通 dict，评测要 json.dumps 它

    def test_encoding_revision_invalidates_both_cache_checks(self):
        """同名模型换了权重版本，也不能加载原索引或复用整库向量。"""
        with tempfile.TemporaryDirectory() as folder:
            before_index = store.input_fingerprint(folder)
            before_vectors = json.dumps(make_cache_key(GOLDEN_HASH, 2), sort_keys=True)
            with patch.dict(EMBEDDING_CONFIG, revision="different-weights"):
                self.assertNotEqual(before_index, store.input_fingerprint(folder))
                self.assertNotEqual(before_vectors, json.dumps(make_cache_key(GOLDEN_HASH, 2), sort_keys=True))


if __name__ == "__main__":
    unittest.main()
