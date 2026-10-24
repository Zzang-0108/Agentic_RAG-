"""真实 LangGraph/工具解析器，假 LLM/检索器：不联网、不加载索引。"""
import json
import unittest
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from rag_core.graph import build_app


class StubLLM(BaseChatModel):
    responses: dict[str, list[Any]] = Field(default_factory=dict)
    calls: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "workflow-stub"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(schema_name=tools[0].__name__)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        name = kwargs.get("schema_name", "text")
        json_mode = name == "text" and "你检查整组资料" in messages[0].content
        if json_mode:
            name = "EvidenceAssessment"
        self.calls.append({"schema": name, "messages": [m.content for m in messages]})
        if not self.responses.get(name):
            raise AssertionError(f"未预期的模型调用：{name}")
        result = self.responses[name].pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, dict):
            message = (AIMessage(content=json.dumps(result, ensure_ascii=False)) if json_mode else
                       AIMessage(content="", tool_calls=[{"name": name, "args": result, "id": "call_test"}]))
        else:
            message = AIMessage(content=result or "没有结构化数据")
        return ChatResult(generations=[ChatGeneration(message=message)])


def route(action="retrieve", question=""):
    return {"action": action, "reason": "测试决策", "clarification_question": question}


def evidence(status="complete", ids=(1,), query="", missing=(), clarification=""):
    return {"status": status, "covered_points": [{"point": "已有要点", "document_ids": list(ids)}] if ids else [],
            "missing_points": list(missing), "followup_query": query,
            "clarification_question": clarification, "reason": "测试依据"}


DOCS = [Document(page_content=t, metadata={"source": f"{i}.pdf", "page_number": i, "book_name": t})
        for i, t in enumerate(("甲", "乙", "丙"), 1)]


class Retriever:
    def __init__(self, batches=None):
        self.batches = list(batches if batches is not None else [DOCS[:2]])
        self.queries = []

    def invoke(self, query, config=None):
        self.queries.append(query)
        batch = self.batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return batch


def run(routes=None, assessments=None, replies=None, batches=None, retries=2, question="原始问题"):
    llm = StubLLM(responses={"RouteDecision": routes or [route()],
                             "EvidenceAssessment": assessments or [], "text": replies or []})
    retriever = Retriever(batches)
    state = build_app(llm, retriever, max_retries=retries).invoke({"question": question})
    return state, llm, retriever


class GraphTests(unittest.TestCase):
    def test_context_budget_applies_to_formatted_context(self):
        from types import SimpleNamespace
        from rag_core.graph import format_docs
        llm = StubLLM(responses={"RouteDecision": [route()],
                                 "EvidenceAssessment": [evidence()], "text": ["答案"]})
        counter = SimpleNamespace(count=len)
        limit = len(format_docs(DOCS[:1]))
        state = build_app(llm, Retriever([DOCS]), token_counter=counter,
                          max_context_tokens=limit).invoke({"question": "专业问题"})
        self.assertEqual(len(state["documents"]), 1)
        self.assertLessEqual(state["trace"][1]["context_budget"]["used_tokens"], limit)

    def test_budget_exhaustion_is_not_misreported_as_empty_search(self):
        from types import SimpleNamespace
        llm = StubLLM(responses={"RouteDecision": [route()]})
        state = build_app(llm, Retriever(), token_counter=SimpleNamespace(count=len),
                          max_context_tokens=1).invoke({"question": "专业问题"})
        self.assertEqual(state["stop_reason"], "context_budget_exhausted")
        self.assertIn("预算", state["generation"])

    def test_direct_uses_no_retrieval(self):
        s, llm, r = run([route("direct")], replies=["43"])
        self.assertEqual(s["generation"], "43")
        self.assertFalse(s["need_retrieve"])
        self.assertEqual(r.queries, [])
        self.assertEqual([c["schema"] for c in llm.calls], ["RouteDecision", "text"])

    def test_clarification_does_not_invent_objects_or_invoke_generation(self):
        s, llm, r = run([route("clarify", "请说明要比较的两个对象。")])
        self.assertEqual(s["generation"], "请说明要比较的两个对象。")
        self.assertEqual(s["route_action"], "clarify")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(r.queries, [])

    def test_missing_object_rule_finishes_without_any_model_call(self):
        s, llm, r = run(question="请帮我对比一下这两个方案。")
        self.assertEqual(s["route_action"], "clarify")
        self.assertEqual(s["final_action"], "clarify")
        self.assertIn("具体对象", s["generation"])
        self.assertEqual(llm.calls, [])
        self.assertEqual(r.queries, [])
        self.assertEqual([event["node"] for event in s["trace"]], ["route", "clarify"])
        self.assertEqual(s["trace"][0]["structured_output"]["source"], "rule")
        self.assertEqual(sum(event["usage"]["llm_started_calls"] for event in s["trace"]), 0)

    def test_new_request_clears_previous_evidence_queries_and_trace(self):
        llm = StubLLM(responses={"RouteDecision": [route(), route("direct")],
                                 "EvidenceAssessment": [evidence()], "text": ["知识答案", "你好"]})
        retriever = Retriever()
        app = build_app(llm, retriever)
        first = app.invoke({"question": "专业问题"})
        second = app.invoke({**first, "question": "你好"})
        self.assertEqual(second["documents"], [])
        self.assertEqual(second["evidence_pool"], [])
        self.assertEqual(second["queries"], [])
        self.assertEqual(second["covered_points"], [])
        self.assertEqual(second["generation"], "你好")
        self.assertEqual([event["node"] for event in second["trace"]], ["route", "direct_answer"])

    def test_whole_context_is_assessed_once(self):
        s, llm, _ = run(assessments=[evidence(ids=(1, 2))], replies=["答案 [资料 1]"])
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(s["final_action"], "answer")
        self.assertEqual(len(s["documents"]), 2)
        self.assertEqual([e["node"] for e in s["trace"]], ["route", "retrieve", "assess", "generate"])
        self.assertEqual(sum(e["usage"]["llm_started_calls"] for e in s["trace"]), 3)

    def test_partial_triggers_targeted_search_and_retains_old_evidence(self):
        s, llm, r = run(assessments=[evidence("partial", ids=(2,), query="缺失信息", missing=("另一部分",)),
                                    evidence(ids=(1, 2))], replies=["完整回答"], batches=[DOCS[:2], [DOCS[2]]])
        self.assertEqual(r.queries, ["原始问题", "缺失信息"])
        self.assertEqual([d.page_content for d in s["documents"]], ["乙", "丙"])
        self.assertEqual(s["retries"], 1)
        prompts = [c["messages"][-1] for c in llm.calls if c["schema"] == "EvidenceAssessment"]
        self.assertTrue(all("原始问题：原始问题" in p for p in prompts))
        self.assertEqual(len(llm.calls), 4)  # 不再单独调 LLM 重写

    def test_evidence_ids_are_remapped_after_filtering(self):
        s, _, _ = run(assessments=[evidence(ids=(2,))], replies=["答案 [资料 1]"])
        self.assertEqual(s["covered_points"][0]["document_ids"], [1])
        self.assertEqual(s["documents"][0].page_content, "乙")

    def test_partial_at_budget_generates_with_explicit_missing_information(self):
        s, llm, _ = run(assessments=[evidence("partial", query="补搜", missing=("缺少 B",))],
                        replies=["A 有依据；B 尚缺资料。"], retries=0)
        self.assertEqual(s["final_action"], "partial_answer")
        self.assertEqual(s["stop_reason"], "retrieval_budget_exhausted")
        self.assertIn("缺少 B", llm.calls[-1]["messages"][0])

    def test_none_at_budget_never_calls_freeform_generator(self):
        s, llm, r = run(assessments=[evidence("none", ids=(), query="补搜", missing=("依据",))], retries=0)
        self.assertEqual(s["final_action"], "abstain")
        self.assertIn("无法充分回答", s["generation"])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(r.queries), 1)

    def test_repeated_query_stops_instead_of_looping(self):
        s, _, r = run(assessments=[evidence("none", ids=(), query=" 原始问题！ ")])
        self.assertEqual(len(r.queries), 1)
        self.assertEqual(s["stop_reason"], "no_new_query")

    def test_retry_budget_is_bounded(self):
        s, _, r = run(assessments=[evidence("none", ids=(), query=q) for q in ("新词甲", "新词乙", "新词丙")],
                      batches=[DOCS[:1]] * 3)
        self.assertEqual(len(r.queries), 3)
        self.assertEqual(s["retries"], 2)
        self.assertEqual(s["final_action"], "abstain")

    def test_missing_external_input_stops_search(self):
        s, _, r = run(assessments=[evidence("needs_input", ids=(), clarification="请提供训练日志。")])
        self.assertEqual(s["final_action"], "clarify")
        self.assertEqual(s["generation"], "请提供训练日志。")
        self.assertEqual(len(r.queries), 1)

    def test_route_text_json_fallback(self):
        raw = '```json\n' + json.dumps({"RouteDecision": route("direct")}) + '\n```'
        s, _, r = run([raw], replies=["你好"])
        self.assertEqual(s["trace"][0]["structured_output"]["source"], "text")
        self.assertEqual(r.queries, [])

    def test_route_failure_defaults_to_retrieval_and_records_error(self):
        s, _, r = run([None, None], assessments=[evidence()], replies=["有证据的答案"])
        self.assertEqual(s["route_action"], "retrieve")
        self.assertEqual(len(r.queries), 1)
        self.assertEqual(s["trace"][0]["structured_output"]["source"], "error")

    def test_invalid_evidence_id_is_retried(self):
        s, _, _ = run(assessments=[evidence(ids=(99,)), evidence(ids=(2,))], replies=["答案"])
        self.assertEqual(s["documents"][0].page_content, "乙")
        self.assertEqual(s["trace"][2]["structured_output"]["attempts"], 2)

    def test_assessment_parse_failure_is_not_irrelevance(self):
        s, llm, _ = run(assessments=[None, None])
        self.assertEqual(s["evidence_status"], "error")
        self.assertEqual(s["stop_reason"], "evidence_check_error")
        self.assertIn("未能完成", s["generation"])
        self.assertEqual(len(s["trace"][1]["pool"]), 2)
        self.assertNotIn("text", [c["schema"] for c in llm.calls])

    def test_retrieval_error_is_not_a_knowledge_absence(self):
        s, _, _ = run(batches=[RuntimeError("模拟网络错误")])
        self.assertEqual(s["stop_reason"], "retrieval_error")
        self.assertIn("未能完成", s["generation"])

    def test_empty_retrieval_never_generates_from_no_evidence(self):
        s, llm, _ = run(batches=[[]])
        self.assertEqual(s["final_action"], "abstain")
        self.assertEqual(len(llm.calls), 1)

    def test_final_state_keeps_evaluator_contract(self):
        s, _, _ = run(assessments=[evidence()], replies=["答案"])
        for k in ("question", "rewritten_query", "documents", "generation", "retries", "need_retrieve"):
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
