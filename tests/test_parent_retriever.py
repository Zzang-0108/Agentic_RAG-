"""父子块检索、预算及落盘兼容：只使用假向量，不调用在线接口。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rag_core.chunking import build_chunks
from rag_core.parent_retriever import ParentContextRetriever, pack_context
from rag_core.provenance import slice_document
from rag_core import store
from rag_core.indexing import build_index, corpus_hash
from tests.test_chunking import Counter, TopicEmbeddings, page, settings


class FakeRetriever:
    k = 5
    def __init__(self, documents):
        self.documents = documents
        self.calls = []
    def invoke(self, query, config=None):
        self.calls.append(query)
        return self.documents


def corpus():
    return build_chunks([page("甲内容及公式。" * 15)], None, Counter(),
                        settings(child_max_tokens=30, embedding_max_chars=30, parent_max_tokens=200))


class ParentTests(unittest.TestCase):
    def test_reranked_children_expand_to_one_parent_with_separate_scores(self):
        bundle = corpus()
        children = bundle.children[:2]
        for i, child in enumerate(children):
            child.metadata["rerank_score"] = 1 - i * .1
        retriever = ParentContextRetriever(base_retriever=FakeRetriever(children), parents=bundle.parents,
                                          token_counter=Counter(), max_context_tokens=1000)
        docs, trace = retriever.retrieve_with_trace("甲")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, next(iter(bundle.parents.values())).page_content)
        self.assertEqual(len(docs[0].metadata["matched_children"]), 2)
        self.assertNotIn("rerank_score", docs[0].metadata)
        self.assertEqual(trace["parent_expansion"]["matched_child_count"], 2)
        self.assertNotIn("matched_children", next(iter(bundle.parents.values())).metadata)

    def test_small_budget_preserves_late_hit_not_the_start_of_parent(self):
        bundle = corpus()
        child = bundle.children[-1]
        retriever = ParentContextRetriever(base_retriever=FakeRetriever([child]), parents=bundle.parents,
                                          token_counter=Counter(), max_context_tokens=35)
        docs, trace = retriever.retrieve_with_trace("甲")
        self.assertEqual(docs[0].page_content, child.page_content)
        self.assertEqual(docs[0].metadata["source_spans"], child.metadata["source_spans"])
        self.assertEqual(docs[0].metadata["chunk_id"], child.metadata["chunk_id"])
        self.assertEqual(trace["parent_expansion"]["child_fallbacks"], 1)
        self.assertLessEqual(trace["parent_expansion"]["used_tokens"], 35)

    def test_expanded_context_budget_counts_list_and_deduplicates(self):
        bundle = corpus()
        retriever = ParentContextRetriever(base_retriever=FakeRetriever(bundle.children[:2]), parents=bundle.parents,
                                          token_counter=Counter())
        docs = retriever.invoke("甲")
        result, budget = pack_context(docs + docs, lambda items: sum(len(d.page_content) + 10 for d in items), 60)
        self.assertTrue(result)
        self.assertLessEqual(budget["used_tokens"], 60)
        self.assertEqual(len({d.metadata["chunk_id"] for d in result}), len(result))

    def test_parent_missing_or_wrong_source_is_explicit_error(self):
        bundle = corpus()
        retriever = ParentContextRetriever(base_retriever=FakeRetriever(bundle.children), parents={}, token_counter=Counter())
        with self.assertRaisesRegex(ValueError, "缺少对应父块"):
            retriever.invoke("甲")
        parent = next(iter(bundle.parents.values()))
        parent.metadata["source"] = "wrong.pdf"
        retriever.parents = bundle.parents
        with self.assertRaisesRegex(ValueError, "不匹配"):
            retriever.invoke("甲")

    def test_provenance_changes_invalidate_new_corpus_hash(self):
        child = corpus().children[0]
        original = corpus_hash([child])
        child.metadata["source_spans"][0]["page_start"] += 1
        self.assertNotEqual(corpus_hash([child]), original)


class OptimizedStoreTests(unittest.TestCase):
    def test_fresh_index_round_trip_and_parent_integrity(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            source = root / "books"
            source.mkdir()
            (source / "test.pdf").write_bytes(b"fingerprint-only")
            config = settings(strategy="semantic")
            backend = TopicEmbeddings()
            cache = root / "cache"
            with patch("file_loader.load_pdf_documents", return_value=[page("甲介绍。甲补充。乙介绍。乙补充。")]):
                fresh = build_index(backend, doc_dir=source, cache_dir=cache, chunking_config=config, token_counter=Counter())
            self.assertFalse(fresh.from_cache)
            self.assertTrue(fresh.parents)
            self.assertTrue((cache / store.PARENTS_NAME).is_file())
            backend.calls.clear()
            restored = build_index(backend, doc_dir=source, cache_dir=cache, chunking_config=config, token_counter=Counter())
            self.assertTrue(restored.from_cache)
            self.assertEqual(backend.calls, [])
            self.assertEqual(restored.parents, fresh.parents)
            (cache / store.PARENTS_NAME).write_text("broken", encoding="utf8")
            self.assertIsNone(store.load(cache, fresh.fingerprint))

    def test_threshold_changes_cache_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            a = store.input_fingerprint(folder, settings(semantic_percentile=90))
            b = store.input_fingerprint(folder, settings(semantic_percentile=95))
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
