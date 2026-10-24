import unittest
from langchain_core.documents import Document
from rag_core.graph import decide_after_grade, decide_retrieve, format_docs, merge_evidence, missing_referent
from rag_core.prompts import EvidenceAssessment


class DecisionTests(unittest.TestCase):
    def test_missing_referent_rule_only_handles_unspecified_requests(self):
        for question in ("请对比这两个模型", "把它和另外那个比较一下。", "继续解释刚才那一段。", "这个参数应该设成多少？"):
            self.assertTrue(missing_referent(question), question)
        for question in ("比较 ResNet 与 MLP", "这个参数 lambda 在 L2 正则化中有什么作用？", "请解释这段文字：梯度下降优化损失"):
            self.assertFalse(missing_referent(question), question)
    def test_three_routes(self):
        for action, node in (("retrieve", "retrieve"), ("direct", "direct_answer"), ("clarify", "clarify")):
            self.assertEqual(decide_retrieve({"route_action": action}), node)

    def test_relevant_document_alone_cannot_trigger_generation(self):
        state = {"documents": [Document(page_content="只回答一部分")], "evidence_status": "partial",
                 "followup_query": "另一部分", "queries": ["原问题"], "retries": 0}
        self.assertEqual(decide_after_grade(state), "rewrite")

    def test_error_stops_without_treating_it_as_no_evidence(self):
        self.assertEqual(decide_after_grade({"evidence_status": "error"}), "abstain")

    def test_merge_preserves_sources_and_enforces_budget(self):
        a = Document(page_content="同文", metadata={"source": "a", "page_number": 1})
        b = Document(page_content="同文", metadata={"source": "b", "page_number": 1})
        self.assertEqual(merge_evidence([a], [a, b], 2), [a, b])
        self.assertEqual(merge_evidence([a], [b], 1), [a])

    def test_complete_without_support_is_invalid(self):
        with self.assertRaises(ValueError):
            EvidenceAssessment(status="complete", covered_points=[], missing_points=[],
                               followup_query="", clarification_question="", reason="")

    def test_format_docs_numbers_the_actual_context(self):
        docs = [Document(page_content="甲", metadata={"book_name": "书一", "page_number": 3}),
                Document(page_content="乙", metadata={})]
        text = format_docs(docs)
        self.assertIn("[资料 1] 书名：书一；PDF页码：3", text)
        self.assertIn("[资料 2]", text)
        self.assertIn("未知", text)
        self.assertEqual(format_docs([]), "")

    def test_cross_page_context_lists_actual_pages_not_just_first_page(self):
        text = format_docs([Document(page_content="跨页资料", metadata={"page_number": 2,
                            "page_numbers": [2, 3], "section_title": "模型评估"})])
        self.assertIn("PDF页码：2,3", text)
        self.assertIn("章节：模型评估", text)


if __name__ == "__main__":
    unittest.main()
