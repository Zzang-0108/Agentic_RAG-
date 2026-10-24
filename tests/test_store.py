"""落盘/读回的往返测试——这里是最容易白花钱的地方，所以测得细一点。

全部在临时目录里跑，不碰 embed_cache_books/，也不会调 embedding 接口。
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from rag_core import store  # noqa: E402
from rag_core.indexing import build_bm25_retriever, build_faiss_index  # noqa: E402


def temp_dir():
    """临时目录。

    ignore_cleanup_errors：Windows 上 faiss/np 读过的文件偶尔还占着句柄，
    清理时报 PermissionError 会让整个用例变成 error。忽略清理报错就行，
    临时目录本来就归系统回收。
    """
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def make_splits():
    return [
        Document(page_content="策略梯度是一种强化学习方法", metadata={"source": "a.pdf", "page_number": 1}),
        Document(page_content="注意力机制用于序列建模", metadata={"source": "b.pdf", "page_number": 2}),
    ]


def make_vectors(n=2, dim=4):
    return np.arange(1, n * dim + 1, dtype=np.float32).reshape(n, dim)


def fake_doc_dir(root):
    """一个空的 PDF 目录——input_fingerprint 不需要真的解析 PDF。"""
    doc_dir = Path(root) / "book"
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir


class TestFingerprint(unittest.TestCase):
    def test_same_input_same_fingerprint(self):
        with temp_dir() as root:
            doc_dir = fake_doc_dir(root)
            (doc_dir / "x.pdf").write_bytes(b"%PDF-1.4 fake")
            self.assertEqual(store.input_fingerprint(doc_dir), store.input_fingerprint(doc_dir))

    def test_changing_a_pdf_changes_the_fingerprint(self):
        with temp_dir() as root:
            doc_dir = fake_doc_dir(root)
            pdf = doc_dir / "x.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake")
            before = store.input_fingerprint(doc_dir)
            pdf.write_bytes(b"%PDF-1.4 fake v2")
            self.assertNotEqual(before, store.input_fingerprint(doc_dir))

    def test_missing_pdf_also_counts(self):
        with temp_dir() as root:
            doc_dir = fake_doc_dir(root)
            pdf = doc_dir / "x.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake")
            before = store.input_fingerprint(doc_dir)
            pdf.unlink()
            self.assertNotEqual(before, store.input_fingerprint(doc_dir))


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self._tmp = temp_dir()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = Path(self._tmp.name) / "cache"
        self.doc_dir = fake_doc_dir(self._tmp.name)
        self.fingerprint = store.input_fingerprint(self.doc_dir)
        self.splits = make_splits()
        self.vectors = make_vectors()
        self.index = build_faiss_index(self.vectors.copy())  # 会原地归一化，所以传副本
        self.bm25 = build_bm25_retriever(self.splits)
        self.cache_key = {"schema_version": 2, "corpus_hash": "deadbeef", "chunk_count": 2,
                          "split_config": {"chunk_size": 500}, "embed_model": "embedding-2"}

    def save(self):
        store.save(self.cache_dir, self.fingerprint, self.cache_key,
                   self.splits, self.vectors, self.index, self.bm25)

    def test_round_trip(self):
        self.save()
        loaded = store.load(self.cache_dir, self.fingerprint)

        self.assertIsNotNone(loaded)
        self.assertEqual([doc.page_content for doc in loaded["splits"]],
                         [doc.page_content for doc in self.splits])
        self.assertEqual(loaded["splits"][0].metadata, self.splits[0].metadata)
        self.assertEqual(loaded["vectors"].shape, self.vectors.shape)
        self.assertEqual(loaded["hnsw_index"].ntotal, 2)
        self.assertEqual(loaded["meta"]["complete"], True)
        self.assertEqual(loaded["meta"]["corpus_hash"], "deadbeef")

    def test_persisted_vectors_are_not_normalized(self):
        """落盘存的是原始向量（和重构前一致），归一化只发生在内存里。"""
        self.save()
        np.testing.assert_allclose(np.load(self.cache_dir / store.VECTORS_NAME), self.vectors)

    def test_bm25_survives_pickling(self):
        """jieba_cut 是模块级具名函数，所以 BM25Retriever 能被 pickle（lambda 不行）。

        断言的是「读回来的检索器行为和原件一致」，不假设具体的排序——
        两篇文档的语料上 BM25 的排序本来就没多大意义。
        """
        self.save()
        loaded = store.load(self.cache_dir, self.fingerprint)

        expected = [doc.page_content for doc in self.bm25.invoke("策略梯度")]
        actual = [doc.page_content for doc in loaded["bm25_retriever"].invoke("策略梯度")]
        self.assertTrue(actual)
        self.assertEqual(actual, expected)

    def test_bundle_completes_only_after_every_file_is_written(self):
        self.save()
        meta = json.loads((self.cache_dir / store.META_NAME).read_text(encoding="utf-8"))
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["input_fingerprint"], self.fingerprint)
        for name in (store.SPLITS_NAME, store.VECTORS_NAME, store.HNSW_NAME, store.BM25_NAME):
            self.assertTrue((self.cache_dir / name).is_file(), name)

    def test_load_returns_none_when_nothing_is_saved(self):
        self.assertIsNone(store.load(self.cache_dir, self.fingerprint))

    def test_load_rejects_a_different_fingerprint(self):
        self.save()
        self.assertIsNone(store.load(self.cache_dir, "some-other-fingerprint"))

    def test_load_rejects_an_incomplete_bundle(self):
        """上次写到一半断电：complete=false 时必须当没缓存，不能拿半截数据凑合。"""
        self.save()
        meta = json.loads((self.cache_dir / store.META_NAME).read_text(encoding="utf-8"))
        meta["complete"] = False
        store.write_meta(self.cache_dir, meta)
        self.assertIsNone(store.load(self.cache_dir, self.fingerprint))

    def test_load_rejects_a_missing_file(self):
        self.save()
        (self.cache_dir / store.HNSW_NAME).unlink()
        self.assertIsNone(store.load(self.cache_dir, self.fingerprint))

    def test_load_rejects_count_mismatch(self):
        """splits 行数和向量行数对不上 -> 缓存作废（宁可重建，不能错位）。"""
        self.save()
        store.write_splits(self.cache_dir / store.SPLITS_NAME, self.splits[:1])
        self.assertIsNone(store.load(self.cache_dir, self.fingerprint))

    def test_load_rejects_nan_vectors(self):
        self.save()
        broken = self.vectors.copy()
        broken[0, 0] = np.nan
        np.save(self.cache_dir / store.VECTORS_NAME, broken)
        self.assertIsNone(store.load(self.cache_dir, self.fingerprint))

    def test_load_rejects_broken_meta_json(self):
        self.save()
        store._write_text_atomic(self.cache_dir / store.META_NAME, "{不是 JSON")
        self.assertIsNone(store.load(self.cache_dir, self.fingerprint))

    def test_broken_bm25_pickle_does_not_kill_the_bundle(self):
        """bm25.pkl 反序列化失败时只该重建 BM25，向量和 HNSW 要继续可用（别重新花钱编码）。"""
        self.save()
        (self.cache_dir / store.BM25_NAME).write_bytes(b"not a pickle at all")

        loaded = store.load(self.cache_dir, self.fingerprint)
        self.assertIsNotNone(loaded)
        self.assertIsNone(loaded["bm25_retriever"])  # 交给 build_index 现场重建
        self.assertEqual(loaded["hnsw_index"].ntotal, 2)

        store.save_bm25(self.cache_dir, self.bm25)
        reloaded = store.load(self.cache_dir, self.fingerprint)
        self.assertIsNotNone(reloaded["bm25_retriever"])

    def test_save_is_repeatable(self):
        self.save()
        self.save()
        self.assertIsNotNone(store.load(self.cache_dir, self.fingerprint))


class TestRuntimeKAlignment(unittest.TestCase):
    """k 是烤进 bm25.pkl 的：TOP_K 改了以后，落盘索引会沿用落盘那天的 k。

    HNSW 那一路每次读回都新建，所以没问题；BM25 是从 pickle 直接取的，会过期。
    缓存命中路径必须把两者对齐，否则两路召回池不等、RRF 融合失衡。
    """

    def setUp(self):
        self._tmp = temp_dir()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = Path(self._tmp.name) / "cache"
        self.doc_dir = fake_doc_dir(self._tmp.name)
        self.fingerprint = store.input_fingerprint(self.doc_dir)
        self.splits = make_splits()
        self.cache_key = {"schema_version": 2, "corpus_hash": "deadbeef", "chunk_count": 2,
                          "split_config": {"chunk_size": 500}, "embed_model": "embedding-2"}

    def save_with_stale_bm25_k(self, stale_k):
        vectors = make_vectors()
        index = build_faiss_index(vectors.copy())
        bm25 = build_bm25_retriever(self.splits)
        bm25.k = stale_k  # 模拟"这份 pickle 是 TOP_K=stale_k 那天落盘的"
        store.save(self.cache_dir, self.fingerprint, self.cache_key,
                   self.splits, vectors, index, bm25)

    def test_cached_bm25_k_is_aligned_to_current_config(self):
        from rag_core.config import TOP_K
        from rag_core.indexing import build_index

        stale_k = TOP_K - 7
        self.assertGreater(stale_k, 0, "测试前提：TOP_K 要大于 7")
        self.save_with_stale_bm25_k(stale_k)

        with contextlib.redirect_stdout(io.StringIO()):
            bundle = build_index(object(), doc_dir=self.doc_dir, cache_dir=self.cache_dir)

        self.assertTrue(bundle.from_cache)
        self.assertEqual(bundle.bm25_retriever.k, TOP_K)
        self.assertEqual(bundle.hnsw_retriever.k, TOP_K)  # 两路池子必须一样宽

    def test_alignment_does_not_rebuild_or_rewrite_the_cache(self):
        """对齐只改内存里的对象，不该顺手改写落盘文件（save 只该由完整构建调用）。"""
        from rag_core.indexing import build_index

        self.save_with_stale_bm25_k(2)
        before = (self.cache_dir / store.BM25_NAME).read_bytes()

        with contextlib.redirect_stdout(io.StringIO()):
            build_index(object(), doc_dir=self.doc_dir, cache_dir=self.cache_dir)

        self.assertEqual((self.cache_dir / store.BM25_NAME).read_bytes(), before)
        # 落盘的还是旧的 k，所以下次读回还会再对齐一次——这是设计上的取舍：
        # 宁可每次多花一行判断，也不为了改一个查询期参数去重写文件。
        self.assertEqual(store.load(self.cache_dir, self.fingerprint)["bm25_retriever"].k, 2)

    def test_hnsw_efsearch_is_aligned_too(self):
        """同上：efSearch 也是构建期烤进索引文件的，读回后要对齐成当前配置。"""
        import faiss

        from rag_core.config import HNSW_EF_SEARCH
        from rag_core.indexing import build_index

        self.save_with_stale_bm25_k(2)
        index_path = self.cache_dir / store.HNSW_NAME
        stale = faiss.read_index(str(index_path))
        stale.hnsw.efSearch = 1  # 一个明显错的搜索深度
        faiss.write_index(stale, str(index_path))

        with contextlib.redirect_stdout(io.StringIO()):
            bundle = build_index(object(), doc_dir=self.doc_dir, cache_dir=self.cache_dir)

        self.assertEqual(bundle.hnsw_index.hnsw.efSearch, HNSW_EF_SEARCH)


    def test_broken_bm25_is_repaired_without_rebuilding_vectors(self):
        from rag_core.indexing import TOP_K, build_index

        self.save_with_stale_bm25_k(2)
        preserved = {
            name: (self.cache_dir / name).read_bytes()
            for name in (store.META_NAME, store.SPLITS_NAME, store.VECTORS_NAME, store.HNSW_NAME)
        }
        (self.cache_dir / store.BM25_NAME).write_bytes(b"broken pickle")

        with patch("file_loader.load_pdf_documents", side_effect=AssertionError("不应解析 PDF")), \
             patch("rag_core.indexing.encode_documents", side_effect=AssertionError("不应编码文档")), \
             contextlib.redirect_stdout(io.StringIO()):
            bundle = build_index(object(), doc_dir=self.doc_dir, cache_dir=self.cache_dir)

        self.assertTrue(bundle.from_cache)
        self.assertEqual(bundle.bm25_retriever.k, TOP_K)
        self.assertIsNotNone(store.load(self.cache_dir, self.fingerprint)["bm25_retriever"])
        for name, content in preserved.items():
            self.assertEqual((self.cache_dir / name).read_bytes(), content, name)

    def test_force_rebuild_loads_documents_and_returns_complete_bundle(self):
        from rag_core.indexing import build_index, corpus_hash

        self.save_with_stale_bm25_k(2)
        embeddings = Mock()
        embeddings.embed_documents.return_value = make_vectors().tolist()
        with patch("file_loader.load_pdf_documents", return_value=make_splits()) as loader, \
             patch("rag_core.indexing.store.load", side_effect=AssertionError("应跳过整库缓存")), \
             contextlib.redirect_stdout(io.StringIO()):
            bundle = build_index(embeddings, doc_dir=self.doc_dir, cache_dir=self.cache_dir,
                                 force_rebuild=True)

        loader.assert_called_once()
        embeddings.embed_documents.assert_called_once()
        self.assertFalse(bundle.from_cache)
        self.assertEqual(bundle.parents, {})
        self.assertEqual(bundle.chunk_count, len(self.splits))
        self.assertEqual(bundle.cache_key["corpus_hash"], corpus_hash(bundle.splits))
        self.assertEqual(bundle.hnsw_index.ntotal, bundle.chunk_count)
        restored = store.load(self.cache_dir, bundle.fingerprint)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(restored["vectors"], make_vectors())


class TestSplitsFile(unittest.TestCase):
    def test_non_json_metadata_is_stringified(self):
        """PyMuPDF 给的 metadata 里可能有 json 存不下的类型（Path 之类），
        default=str 兜底，不能直接抛。"""
        with temp_dir() as root:
            path = Path(root) / "splits.jsonl"
            splits = [Document(page_content="甲", metadata={"source_path": Path("knowledge_db/book/a.pdf")})]
            store.write_splits(path, splits)
            back = store.read_splits(path)
            self.assertEqual(back[0].page_content, "甲")
            self.assertIsInstance(back[0].metadata["source_path"], str)


if __name__ == "__main__":
    unittest.main()
