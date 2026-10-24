"""RerankingRetriever / dedupe / 打分器：全用假的 scorer 和假的基检索器。

不起模型、不联网、不花钱，一秒内跑完。真实的交叉编码器只由
tests 之外的手工验证覆盖（权重加载必须有网/有缓存，不适合放单测里）。
"""
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.documents import Document  # noqa: E402

from rag_core.rerank import (  # noqa: E402
    CrossEncoderScorer,
    RerankingRetriever,
    build_reranking_retriever,
    dedupe_documents,
    logits_to_scores,
    make_scorer,
)


def doc(text, source="a.pdf", page=1, **extra):
    return Document(page_content=text, metadata={"source": source, "page_number": page, **extra})


class FakeBaseRetriever:
    """只实现 invoke()，够 RerankingRetriever 用。"""

    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = []

    def invoke(self, query, config=None):
        self.calls.append(query)
        return list(self.documents)


class FixedScorer:
    """按正文查表返回写死的分数，用来构造可预期的排序。"""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [self.table[text] for text in passages]


class ExplodingScorer:
    """如果被调用就说明不该调用它。"""

    def score(self, query, passages):
        raise AssertionError("scorer 不应该被调用")


# ========== 排序与截断 ==========
class RerankingTests(unittest.TestCase):
    def test_trace_keeps_candidates_and_selected_order_per_request(self):
        retriever = RerankingRetriever(base_retriever=FakeBaseRetriever([doc("甲"), doc("乙")]),
                                      scorer=FixedScorer({"甲": .1, "乙": .9}), top_n=1)
        out, trace = retriever.retrieve_with_trace("问题")
        self.assertEqual([d.page_content for d in out], ["乙"])
        self.assertEqual([d["text"] for d in trace["candidates"]], ["甲", "乙"])
        self.assertEqual(trace["selected"][0]["rerank_score"], .9)
        self.assertGreaterEqual(trace["rerank_ms"], 0)

    def test_sorts_by_score_not_by_recall_order(self):
        """召回顺序是 [甲,乙,丙]，重排后必须是 [乙,丙]——前两名正好是召回排在后面的两个。"""
        base = FakeBaseRetriever([doc("甲"), doc("乙"), doc("丙")])
        retriever = RerankingRetriever(
            base_retriever=base, scorer=FixedScorer({"甲": 0.1, "乙": 0.9, "丙": 0.8}), top_n=2
        )

        out = retriever.invoke("问题")

        self.assertEqual([d.page_content for d in out], ["乙", "丙"])
        self.assertEqual(base.calls, ["问题"])

    def test_top_n_limits_the_result(self):
        documents = [doc(f"第{i}段") for i in range(10)]
        base = FakeBaseRetriever(documents)
        scorer = FixedScorer({d.page_content: float(i) for i, d in enumerate(documents)})
        retriever = RerankingRetriever(base_retriever=base, scorer=scorer, top_n=3)

        out = retriever.invoke("问题")

        self.assertEqual(len(out), 3)
        self.assertEqual([d.page_content for d in out], ["第9段", "第8段", "第7段"])

    def test_scores_are_written_but_cosine_score_is_kept(self):
        """两种分数并存：metadata["score"] 是 HNSW 写的余弦，rerank_score 是重排分。"""
        base = FakeBaseRetriever([doc("甲", score=0.81), doc("乙", score=0.42)])
        retriever = RerankingRetriever(
            base_retriever=base, scorer=FixedScorer({"甲": 0.7, "乙": 0.95}), top_n=2
        )

        out = retriever.invoke("问题")

        self.assertEqual(out[0].page_content, "乙")
        self.assertAlmostEqual(out[0].metadata["rerank_score"], 0.95)
        self.assertAlmostEqual(out[0].metadata["score"], 0.42)  # 没被覆盖

    def test_does_not_touch_the_shared_document_metadata(self):
        """基检索器返回的 Document 和 splits 里的是同一批对象，不能就地写 rerank_score。

        rerank_score 依赖当前 query，写进去比 metadata["score"] 更容易污染落盘的
        splits.jsonl——所以是 model_copy 换一份 metadata，不是就地赋值。
        """
        shared = doc("甲")
        base = FakeBaseRetriever([shared, doc("乙")])
        retriever = RerankingRetriever(
            base_retriever=base, scorer=FixedScorer({"甲": 0.9, "乙": 0.1}), top_n=2
        )

        out = retriever.invoke("问题")

        self.assertIn("rerank_score", out[0].metadata)
        self.assertNotIn("rerank_score", shared.metadata)
        self.assertIsNot(out[0], shared)  # 返回的是副本

    def test_scorer_gets_the_query_and_the_passages_in_order(self):
        base = FakeBaseRetriever([doc("甲"), doc("乙")])
        scorer = FixedScorer({"甲": 0.5, "乙": 0.5})
        retriever = RerankingRetriever(base_retriever=base, scorer=scorer, top_n=2)

        retriever.invoke("原始问题")

        query, passages = scorer.calls[0]
        self.assertEqual(query, "原始问题")
        self.assertEqual(passages, ["甲", "乙"])

    def test_ties_keep_recall_order(self):
        """分数一样时按召回顺序，不许出现随机顺序。"""
        base = FakeBaseRetriever([doc("甲"), doc("乙"), doc("丙")])
        retriever = RerankingRetriever(
            base_retriever=base, scorer=FixedScorer({"甲": 0.5, "乙": 0.5, "丙": 0.5}), top_n=3
        )

        out = retriever.invoke("问题")

        self.assertEqual([d.page_content for d in out], ["甲", "乙", "丙"])


# ========== 关掉重排 ==========
class DisabledTests(unittest.TestCase):
    """RAG_RERANK=0 的语义：真·关掉，行为必须和不加重排时一模一样。"""

    def test_factory_returns_the_base_retriever_untouched(self):
        """不做任何包装、不截断、不写 rerank_score——连一个多余的影响都不留。"""
        base = FakeBaseRetriever([doc(f"第{i}段") for i in range(30)])

        out = build_reranking_retriever(base, scorer=None)

        self.assertIs(out, base)
        self.assertEqual(len(out.invoke("问题")), 30)  # 条数也和以前一样

    def test_make_scorer_disabled_returns_none(self):
        self.assertIsNone(make_scorer(enabled=False))

    def test_make_scorer_does_not_load_the_model(self):
        """造对象和加载权重是两件事：import rag 之后、warmup 之前不该占显存。"""
        scorer = make_scorer(enabled=True)
        self.assertFalse(scorer.loaded)
        self.assertEqual(scorer.load_seconds, 0.0)

    def test_make_scorer_passes_explicit_flag_through(self):
        """显式传 False 时不该去读 RERANK_ENABLED。"""
        self.assertIsInstance(make_scorer(enabled=True), CrossEncoderScorer)


# ========== 去重 ==========
class DedupeTests(unittest.TestCase):
    def test_identical_chunks_collapse(self):
        documents = [doc("甲"), doc("甲"), doc("乙"), doc("甲")]
        self.assertEqual([d.page_content for d in dedupe_documents(documents)], ["甲", "乙"])

    def test_same_text_on_different_books_is_kept(self):
        """EnsembleRetriever 只按正文去重，跨书的相同片段会被吃掉，这里不会。"""
        documents = [doc("同样的正文", source="a.pdf", page=1),
                     doc("同样的正文", source="b.pdf", page=7)]
        self.assertEqual(len(dedupe_documents(documents)), 2)

    def test_same_text_on_different_pages_is_kept(self):
        documents = [doc("同样的正文", page=1), doc("同样的正文", page=2)]
        self.assertEqual(len(dedupe_documents(documents)), 2)

    def test_order_is_preserved(self):
        documents = [doc("丙"), doc("甲"), doc("乙")]
        self.assertEqual([d.page_content for d in dedupe_documents(documents)], ["丙", "甲", "乙"])

    def test_dedupe_happens_before_scoring(self):
        base = FakeBaseRetriever([doc("甲"), doc("甲"), doc("乙")])
        scorer = FixedScorer({"甲": 0.9, "乙": 0.1})
        retriever = RerankingRetriever(base_retriever=base, scorer=scorer, top_n=5)

        out = retriever.invoke("问题")

        self.assertEqual(len(out), 2)
        self.assertEqual(scorer.calls[0][1], ["甲", "乙"])  # 没有重复打分


# ========== 边界 ==========
class EdgeCaseTests(unittest.TestCase):
    def test_empty_candidates(self):
        retriever = RerankingRetriever(
            base_retriever=FakeBaseRetriever([]), scorer=ExplodingScorer(), top_n=5
        )
        self.assertEqual(retriever.invoke("问题"), [])

    def test_single_candidate_is_still_scored(self):
        """只有一条时也打分——不然 rerank_score 会时有时无，日志读起来莫名其妙。"""
        base = FakeBaseRetriever([doc("甲")])
        scorer = FixedScorer({"甲": 0.63})
        retriever = RerankingRetriever(base_retriever=base, scorer=scorer, top_n=5)

        out = retriever.invoke("问题")

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].metadata["rerank_score"], 0.63)

    def test_fewer_candidates_than_top_n(self):
        base = FakeBaseRetriever([doc("甲"), doc("乙")])
        retriever = RerankingRetriever(
            base_retriever=base, scorer=FixedScorer({"甲": 0.2, "乙": 0.8}), top_n=20
        )
        self.assertEqual(len(retriever.invoke("问题")), 2)

    def test_k_reports_top_n_not_the_recall_pool(self):
        """evaluate.py 读 .k 记进报告；对外承诺的是最终条数，不是召回池大小。"""
        retriever = RerankingRetriever(
            base_retriever=FakeBaseRetriever([]), scorer=FixedScorer({}), top_n=7
        )
        self.assertEqual(retriever.k, 7)


# ========== 工厂 ==========
class FactoryTests(unittest.TestCase):
    def test_factory_uses_the_given_top_n(self):
        retriever = build_reranking_retriever(
            FakeBaseRetriever([]), scorer=FixedScorer({}), top_n=9
        )
        self.assertIsInstance(retriever, RerankingRetriever)
        self.assertEqual(retriever.top_n, 9)
        self.assertEqual(retriever.k, 9)

    def test_factory_defaults_top_n_to_config(self):
        from rag_core.config import RERANK_TOP_N

        retriever = build_reranking_retriever(FakeBaseRetriever([]), scorer=FixedScorer({}))
        self.assertEqual(retriever.top_n, RERANK_TOP_N)


# ========== 打分后的后处理 ==========
class LogitsTests(unittest.TestCase):
    """纯 Python，不 import torch——测试套件要保持在 1 秒级。"""

    def test_single_logit_goes_through_sigmoid(self):
        scores = logits_to_scores([[0.0], [4.0], [-4.0]])

        self.assertAlmostEqual(scores[0], 0.5, places=6)
        self.assertGreater(scores[1], 0.97)
        self.assertLess(scores[2], 0.03)

    def test_extreme_values_do_not_overflow(self):
        """sigmoid 不溢出：分正负两支就是为了这个。"""
        scores = logits_to_scores([[800.0], [-800.0]])

        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 0.0)

    def test_two_logits_take_the_positive_class(self):
        scores = logits_to_scores([[-2.0, 2.0], [2.0, -2.0]])

        self.assertGreater(scores[0], 0.97)  # 正类在后
        self.assertLess(scores[1], 0.03)

    def test_softmax_branch_does_not_overflow(self):
        scores = logits_to_scores([[0.0, 900.0]])

        self.assertAlmostEqual(scores[0], 1.0, places=6)

    def test_order_is_preserved(self):
        """分数必须和传入的候选一一对应，不能乱序。"""
        scores = logits_to_scores([[0.0], [1.0], [2.0]])
        self.assertEqual(scores, sorted(scores))

    def test_empty_input(self):
        self.assertEqual(logits_to_scores([]), [])

    def test_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            logits_to_scores([[1.0, 2.0, 3.0]])

    def test_returns_plain_floats(self):
        self.assertIsInstance(logits_to_scores([[1.0]])[0], float)


# ========== 打分器本身（不加载权重） ==========
class ScorerTests(unittest.TestCase):
    def test_score_with_no_passages_returns_empty_without_loading(self):
        """空候选必须提前返回：否则会为了打 0 条候选去加载 1.1GB 权重。"""
        scorer = CrossEncoderScorer()
        self.assertEqual(scorer.score("问题", []), [])
        self.assertFalse(scorer.loaded)

    def test_warmup_is_idempotent(self):
        scorer = CrossEncoderScorer()
        scorer.load_seconds = 4.2
        scorer._model = SimpleNamespace()  # 假装已经加载好
        self.assertEqual(scorer.warmup(), 4.2)


class ImportDisciplineTests(unittest.TestCase):
    def test_importing_the_module_does_not_pull_in_torch(self):
        """模块级不许 import torch / transformers。

        否则 RAG_RERANK=0 也省不掉那几秒，而且测试套件会被拖慢好几秒。
        放在子进程里验：进程内 sys.modules 已经被别的用例污染，测不准。
        """
        code = (
            "import sys; import rag_core.rerank; "
            "print(','.join(m for m in ('torch', 'transformers') if m in sys.modules))"
        )
        finished = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(finished.stdout.strip(), "", "rag_core.rerank 在模块级把重家伙 import 了")


if __name__ == "__main__":
    unittest.main()
