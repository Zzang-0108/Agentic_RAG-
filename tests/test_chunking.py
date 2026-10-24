"""合成文档功能测试：不读取评测题、不调用模型接口。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from langchain_core.documents import Document
from rag_core.chunking import build_chunks, sentence_units
from rag_core.config import CHUNKING_CONFIG
from rag_core.embedding_cache import CachedEmbeddings
from rag_core.provenance import slice_document, validate_provenance
from rag_core.structure import build_sections


class Counter:
    model_name = "test-character-counter"
    signature = "test-v1"
    count = staticmethod(len)


class TopicEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0., 1.] if "乙" in text else [1., 0.] for text in texts]


def page(text, number=1, source="a.pdf"):
    return Document(page_content=text, metadata={"source": source, "page_number": number, "book_name": source})


def settings(**kwargs):
    return {**CHUNKING_CONFIG, "strategy": "structural", "child_max_tokens": 80,
            "child_min_tokens": 0, "parent_max_tokens": 300, "embedding_max_chars": 80,
            "semantic_buffer": 0, "semantic_percentile": 50, **kwargs}


class StructureTests(unittest.TestCase):
    def test_cross_page_continuation_and_section_boundary_keep_exact_provenance(self):
        pages = [page("第1章 测试\n这是跨页的定", 1),
                 page("义，以及公式的说明。\n1.1 下一节\n这是另一节的内容。", 2)]
        sections, _ = build_sections(pages)
        self.assertEqual(len(sections), 2)
        self.assertIn("跨页的定义", sections[0].page_content)
        self.assertEqual(sections[0].metadata["page_numbers"], [1, 2])
        self.assertEqual(sections[1].metadata["page_numbers"], [2])
        raw = {d.metadata["page_number"]: d.page_content for d in pages}
        for section in sections:
            validate_provenance(section)
            for span in section.metadata["source_spans"]:
                self.assertEqual(section.page_content[span["start"]:span["end"]],
                                 raw[span["page_number"]][span["page_start"]:span["page_end"]])

    def test_different_books_and_missing_pages_do_not_merge(self):
        sections, _ = build_sections([page("甲正文。", 1), page("乙正文。", 3), page("丙正文。", 1, "b.pdf")])
        self.assertEqual(len(sections), 3)

    def test_repeated_headers_removed_without_collapsing_chapter_numbers(self):
        pages = [page(f"教程页眉\n第{i}章\n正文内容说明。\n{i}", i) for i in range(1, 5)]
        sections, stats = build_sections(pages)
        text = "".join(s.page_content for s in sections)
        for i in range(1, 5):
            self.assertIn(f"第{i}章", text)
        self.assertEqual(text.count("教程页眉"), 0)
        self.assertEqual(text.count("正文内容说明。"), 4)
        self.assertEqual(stats["repeated_headers_removed"], 4)

    def test_bookmark_and_split_number_heading(self):
        pages = [page("2.2\n评估方法\n交叉验证介绍。\n附加阅读\n补充内容。")]
        sections, _ = build_sections(pages, {("a.pdf", 1): [(2, "附加阅读")]})
        self.assertEqual(len(sections), 2)
        self.assertIn("2.2 评估方法", sections[0].metadata["section_title"])
        self.assertTrue(sections[1].metadata["section_title"].endswith("附加阅读"))

    def test_split_heading_and_bookmark_do_not_create_number_only_chunks(self):
        pages = [page("第1章\n绪论\n1.1\n强化学习概述\n正文说明。")]
        sections, _ = build_sections(pages, {("a.pdf", 1): [(1, "绪论"), (2, "强化学习概述")]})
        self.assertEqual(len(sections), 1)
        self.assertIn("正文说明", sections[0].page_content)
        self.assertIn("强化学习概述", sections[0].metadata["section_title"])

    def test_spaced_toc_leaders_are_not_treated_as_real_headings(self):
        pages = [page("目录\n1.1 引言 . . . . 3\n1.2 定义 . . . . 4\n1.3 习题 . . . . 5"),
                 page("第1章\n绪论\n正文说明。", 2)]
        sections, stats = build_sections(pages)
        self.assertEqual(stats["toc_pages_removed"], 1)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].metadata["page_numbers"], [2])

    def test_slice_only_claims_pages_actually_contained(self):
        sections, _ = build_sections([page("第一部分。", 1), page("第二部分。", 2)])
        section = sections[0]
        start = section.page_content.index("第二")
        child = slice_document(section, start, len(section.page_content))
        self.assertEqual(child.metadata["page_numbers"], [2])
        self.assertEqual(child.metadata["source_spans"][0]["start"], 0)

    def test_table_decimal_values_are_not_numbered_headings(self):
        text = "2.2 数据预处理\n正文介绍。\nNumRooms Alley\n0\n3.0\nPave\n1\n2.0\nNaN\n2\n4.0\nNaN\n3\n3.0\nNaN\n正文解释。"
        sections, _ = build_sections([page(text)])
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].metadata["section_title"], "2.2 数据预处理")
        self.assertIn("Pave", sections[0].page_content)
        self.assertEqual(sections[0].page_content.count("NaN"), 3)

    def test_decorative_arrows_removed_inside_pdf_reading_order(self):
        sections, _ = build_sections([page("正文开头。\n第一段内容。\n←_←\n第二段内容。\n正文结束。")])
        self.assertNotIn("←_←", sections[0].page_content)
        self.assertIn("第二段内容。", sections[0].page_content)


class ChunkingTests(unittest.TestCase):
    def test_semantic_boundaries_change_groups_without_losing_text(self):
        original = "甲话题内容。甲补充说明。乙话题内容。乙补充说明。"
        embeddings = TopicEmbeddings()
        semantic = build_chunks([page(original)], embeddings, Counter(), settings(strategy="semantic"))
        structural = build_chunks([page(original)], None, Counter(), settings())
        self.assertEqual(len(structural.children), 1)
        self.assertEqual(len(semantic.children), 2)
        self.assertEqual("".join(d.page_content for d in semantic.children), original)
        self.assertEqual(semantic.stats["semantic_boundaries"], 1)
        self.assertEqual(len(embeddings.calls), 1)

    def test_semantic_error_does_not_silently_fallback(self):
        with self.assertRaises(ValueError):
            build_chunks([page("甲。乙。")], None, Counter(), settings(strategy="semantic"))
        bad = SimpleNamespace(embed_documents=lambda texts: [[0., 0.]] * len(texts))
        with self.assertRaisesRegex(ValueError, "零向量"):
            build_chunks([page("甲。乙。")], bad, Counter(), settings(strategy="semantic"))

    def test_lengths_parent_links_and_ids_are_deterministic(self):
        pages = [page("甲话题的连续文字。" * 30)]
        config = settings(child_max_tokens=30, parent_max_tokens=100, embedding_max_chars=25)
        bundle = build_chunks(pages, None, Counter(), config)
        again = build_chunks(pages, None, Counter(), config)
        self.assertEqual([d.metadata for d in bundle.children], [d.metadata for d in again.children])
        for child in bundle.children:
            self.assertLessEqual(len(child.page_content), 25)
            parent = bundle.parents[child.metadata["parent_id"]]
            self.assertLessEqual(len(parent.page_content), 100)
            self.assertEqual(parent.page_content[child.metadata["parent_start"]:child.metadata["parent_end"]], child.page_content)

    def test_fenced_code_and_formula_stay_atomic_when_they_fit(self):
        text = '文字。\n```python\nx = 3.14\nprint("乙。")\n```\n$$a=b+c$$\n解释。'
        units = sentence_units(text, len, 100)
        pieces = [text[a:b] for a, b, _ in units]
        self.assertIn('```python\nx = 3.14\nprint("乙。")\n```', pieces)
        self.assertIn("$$a=b+c$$", pieces)
        self.assertFalse(any(forced for _, _, forced in units))

    def test_large_atomic_block_is_bounded_and_marked_not_dropped(self):
        text = "$$" + "x" * 100 + "$$"
        units = sentence_units(text, len, 20)
        self.assertEqual("".join(text[a:b] for a, b, _ in units), text)
        self.assertTrue(all(forced for _, _, forced in units))
        self.assertTrue(all(b - a <= 20 for a, b, _ in units))

    def test_no_spurious_break_for_identical_embeddings(self):
        bundle = build_chunks([page("甲。甲。甲。")], TopicEmbeddings(), Counter(), settings(strategy="semantic"))
        self.assertEqual(bundle.stats["semantic_boundaries"], 0)


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_deduplicates_and_persists_without_mixing_models(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.sqlite"
            backend = TopicEmbeddings()
            cache = CachedEmbeddings(backend, path, "model-a", 2)
            result = cache.embed_documents(["甲", "乙", "甲"])
            self.assertEqual(backend.calls, [["甲", "乙"]])
            self.assertEqual(result[0], result[2])
            backend.calls.clear()
            CachedEmbeddings(backend, path, "model-a").embed_documents(["乙", "甲"])
            self.assertEqual(backend.calls, [])
            CachedEmbeddings(backend, path, "model-b").embed_documents(["甲"])
            self.assertEqual(backend.calls, [["甲"]])

    def test_completed_batches_survive_later_api_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.sqlite"
            def encode(texts):
                if texts == ["乙"]:
                    raise RuntimeError("模拟接口中断")
                return [[1., 0.]]
            cache = CachedEmbeddings(SimpleNamespace(embed_documents=encode), path, "model", 1)
            with self.assertRaises(RuntimeError):
                cache.embed_documents(["甲", "乙"])
            backend = TopicEmbeddings()
            CachedEmbeddings(backend, path, "model", 1).embed_documents(["甲", "乙"])
            self.assertEqual(backend.calls, [["乙"]])


if __name__ == "__main__":
    unittest.main()
