"""三路路由 → 重排检索 → 整组证据检查 → 按缺失点补搜或回答。"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langgraph.graph import StateGraph, END

from .config import MAX_RETRIES, MAX_EVIDENCE_DOCS
from .prompts import (DIRECT_PROMPT, GENERATE_PROMPT, EVIDENCE_PROMPT, ROUTE_PROMPT,
                      EvidenceAssessment, RouteDecision)
from .structured import invoke_structured
from .telemetry import NodeUsage, dedupe_documents, document_id, document_snapshot
from .provenance import page_numbers
from .parent_retriever import pack_context


class GraphState(TypedDict, total=False):
    question: str
    rewritten_query: str
    documents: list[Document]  # 实际交给生成节点的资料；评测使用 final_context
    evidence_pool: list[Document]
    generation: str
    retries: int
    need_retrieve: bool  # 保留旧评测接口；三路动作另见 route_action
    route_action: str
    final_action: str
    clarification_question: str
    evidence_status: str
    covered_points: list[dict]
    missing_points: list[str]
    followup_query: str
    queries: list[str]
    stop_reason: str
    trace: list[dict]


def format_docs(docs: list[Document]) -> str:
    titles = {"pumpkin_book": "南瓜书", "d2l-zh-pytorch": "动手学深度学习",
              "EasyRL_v1.0.6": "Easy RL", "v1": "动手学大模型应用开发"}
    sections = []
    for number, doc in enumerate(docs, start=1):
        book = doc.metadata.get("book_name", "未知")
        title = titles.get(book, book)
        pages = ",".join(map(str, page_numbers(doc.metadata))) or "未知"
        header = f"[资料 {number}] 书名：{title}；PDF页码：{pages}"
        if doc.metadata.get("section_title"):
            header += "；章节：" + doc.metadata["section_title"]
        sections.append(header + "\n" + doc.page_content)
    return "\n\n---\n\n".join(sections)


def query_key(query):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", query).casefold())


def missing_referent(question):
    """仅拦截没有具体对象的短请求；出现实际术语/对象后交给模型，不按题号匹配。"""
    text = query_key(question)
    if re.search("比较|对比", text):
        remaining = re.sub("比较|对比|请|帮我|麻烦|把|将|一下|看看|它们|它|这个|那个|这|那|另外|另一个|两个|个|和|与|跟|模型|方案|结果", "", text)
        if not remaining:
            return "请说明要比较的具体对象，以及希望比较哪些方面。"
    if re.fullmatch(r"(?:请|帮我|继续)*(?:解释|总结|分析|讲讲)(?:刚才|之前|前面|上面|那|这)+(?:的|一|个|段|句话|内容|文字)*", text):
        return "请提供要处理的原文或前文，这条消息里还没有具体内容。"
    if re.fullmatch(r"(?:请|帮我|告诉我)*(?:这个|那个|这项|该)参数(?:应该|应当|该|要)?(?:设成|设置为|设为|设置|取)(?:多少|什么值)", text):
        return "请说明参数名称、使用的模型或任务，以及相关约束。"
    return ""


def merge_evidence(previous, incoming, limit):
    """保留已确认有用的旧证据；来源/页码/正文联合去重，按明确预算截断。"""
    return dedupe_documents([*previous, *incoming])[:limit]


def decide_retrieve(state: GraphState) -> str:
    if "route_action" in state:
        action = state["route_action"]
    else:
        action = "retrieve" if state.get("need_retrieve") else "direct"
    return {"retrieve": "retrieve", "direct": "direct_answer", "clarify": "clarify"}[action]


def decide_after_grade(state: GraphState, max_retries: int = MAX_RETRIES) -> str:
    """保留函数名供调用方使用，判断依据改为整组证据的状态。"""
    status = state.get("evidence_status", "error")
    if status == "needs_input":
        return "clarify"
    if status == "complete":
        return "generate"
    if status == "error":
        return "abstain"
    next_query = query_key(state.get("followup_query", "").strip())
    seen = {query_key(q) for q in state.get("queries", [])}
    if state.get("retries", 0) < max_retries and next_query and next_query not in seen:
        return "rewrite"
    return "generate" if status == "partial" and state.get("documents") else "abstain"


def build_app(llm, retriever, max_retries: int = MAX_RETRIES,
              max_evidence_docs: int = MAX_EVIDENCE_DOCS, token_counter=None,
              max_context_tokens: int = 4096):
    if max_retries < 0 or max_evidence_docs < 1:
        raise ValueError("max_retries 必须非负，max_evidence_docs 必须为正")
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens 必须为正")
    route_chain = ROUTE_PROMPT | llm.with_structured_output(RouteDecision, include_raw=True)
    # 长上下文下当前生成模型容易复述工具定义；直接要求 JSON，再用同一 schema 严格校验。
    checker = EVIDENCE_PROMPT | llm
    generate_chain = GENERATE_PROMPT | llm | StrOutputParser()
    direct_chain = DIRECT_PROMPT | llm | StrOutputParser()

    def measured(name, function):
        def node(state: GraphState, config: RunnableConfig):
            tracker = NodeUsage()
            started = time.perf_counter()
            updates, detail = function(state, merge_configs(config, {"callbacks": [tracker]}))
            event = {"node": name, "round": updates.get("retries", state.get("retries", 0)),
                     "latency_ms": (time.perf_counter() - started) * 1000,
                     "usage": tracker.snapshot(), **detail}
            updates["trace"] = ([] if name == "route" else state.get("trace", [])) + [event]
            return updates
        return node

    def route_question(state, config):
        clarification = missing_referent(state["question"])
        if clarification:
            result = RouteDecision(action="clarify", reason="单轮请求没有提供具体指代对象", clarification_question=clarification)
            diagnostic = {"source": "rule", "attempts": 0, "errors": []}
        else:
            result, diagnostic = invoke_structured(route_chain, {"question": state["question"]},
                                                   RouteDecision, config)
        action = result.action if result else "retrieve"
        reason = result.reason if result else "路由解析失败；保守进入检索，不直接编写知识答案"
        updates = {"route_action": action, "need_retrieve": action == "retrieve",
                   "rewritten_query": state["question"], "retries": 0, "queries": [],
                   "documents": [], "evidence_pool": [], "generation": "",
                   "covered_points": [], "missing_points": [], "evidence_status": "unchecked",
                   "followup_query": "", "stop_reason": "", "final_action": "",
                   "clarification_question": result.clarification_question if result else ""}
        return updates, {"action": action, "reason": reason, "structured_output": diagnostic}

    def retrieve(state, config):
        query = state["rewritten_query"]
        try:
            if hasattr(retriever, "retrieve_with_trace"):
                docs, detail = retriever.retrieve_with_trace(query, config=config)
            else:
                docs = list(retriever.invoke(query, config=config))
                detail = {"selected": [document_snapshot(d, i) for i, d in enumerate(docs, 1)]}
            pool = merge_evidence(state["documents"], docs, max_evidence_docs)
            budget = None
            if token_counter is not None:
                pool, budget = pack_context(pool, lambda items: token_counter.count(format_docs(items)),
                                             max_context_tokens)
            return {"evidence_pool": pool, "queries": state["queries"] + [query],
                    "evidence_status": "unchecked",
                    "stop_reason": "context_budget_exhausted" if docs and not pool and budget else ""}, {
                        "query": query, "retrieval": detail, "context_budget": budget,
                        "pool": [document_snapshot(d, i) for i, d in enumerate(pool, 1)]}
        except Exception as exc:
            return {"evidence_status": "error", "stop_reason": "retrieval_error",
                    "queries": state["queries"] + [query]}, {
                        "query": query, "error": {"type": type(exc).__name__, "message": str(exc)[:500]}}

    def assess(state, config):
        if state["evidence_status"] == "error":
            return {}, {"status": "error", "reason": "检索调用失败，不作证据缺失结论"}
        pool = state["evidence_pool"]
        if not pool:
            budget_exhausted = state.get("stop_reason") == "context_budget_exhausted"
            reason = "检索结果超出当前上下文预算" if budget_exhausted else "没有检索到可供核对的资料"
            return {"evidence_status": "none", "documents": [], "followup_query": "",
                    "missing_points": [reason],
                    "stop_reason": "context_budget_exhausted" if budget_exhausted else "empty_retrieval"}, {
                        "status": "none", "reason": reason}

        def valid_ids(result):
            for point in result.covered_points:
                if any(i < 1 or i > len(pool) for i in point.document_ids):
                    raise ValueError("证据编号超出本轮资料范围")

        result, diagnostic = invoke_structured(
            checker, {"question": state["question"], "queries": json.dumps(state["queries"], ensure_ascii=False),
                      "context": format_docs(pool)}, EvidenceAssessment, config, validate=valid_ids)
        if result is None:
            # 保留上一轮已确认的资料供排错；不把未能判断的新资料标为不相关。
            return {"evidence_status": "error", "stop_reason": "evidence_check_error"}, {
                        "status": "error", "structured_output": diagnostic}
        used = {i for p in result.covered_points for i in p.document_ids}
        selected = [doc for i, doc in enumerate(pool, 1) if i in used]
        remap = {old: new for new, old in enumerate(sorted(used), 1)}
        covered = [{"point": p.point, "document_ids": [remap[i] for i in p.document_ids]}
                   for p in result.covered_points]
        updates = {"evidence_status": result.status, "documents": selected, "covered_points": covered,
                   "missing_points": result.missing_points, "followup_query": result.followup_query.strip(),
                   "clarification_question": result.clarification_question, "stop_reason": ""}
        following = decide_after_grade({**state, **updates}, max_retries)
        if following != "rewrite":
            if result.status in ("complete", "needs_input"):
                updates["stop_reason"] = result.status
            elif state["retries"] >= max_retries:
                updates["stop_reason"] = "retrieval_budget_exhausted"
            else:
                updates["stop_reason"] = "no_new_query"
        return updates, {"assessment": result.model_dump(), "structured_output": diagnostic,
                         "kept_chunk_ids": [document_id(doc) for doc in selected], "next": following,
                         "stop_reason": updates["stop_reason"]}

    def rewrite(state, config):
        # 查询已由整组检查针对缺失点产生，不再为同一件事额外调用一次模型。
        return {"rewritten_query": state["followup_query"], "retries": state["retries"] + 1}, {
                    "query": state["followup_query"], "missing_points": state["missing_points"]}

    def generate(state, config):
        answer = generate_chain.invoke({"question": state["question"], "context": format_docs(state["documents"]),
                                        "evidence_status": state["evidence_status"],
                                        "missing_points": json.dumps(state["missing_points"], ensure_ascii=False)},
                                       config=config).strip()
        if not answer:
            raise ValueError("生成节点未返回非空答案")
        action = "answer" if state["evidence_status"] == "complete" else "partial_answer"
        return {"generation": answer, "final_action": action}, {
                    "action": action, "context_chunk_ids": [document_id(d) for d in state["documents"]]}

    def direct_answer(state, config):
        answer = direct_chain.invoke({"question": state["question"]}, config=config).strip()
        if not answer:
            raise ValueError("直接回答节点未返回非空答案")
        return {"generation": answer, "final_action": "direct", "stop_reason": "direct"}, {"action": "direct"}

    def clarify(state, config):
        answer = state["clarification_question"].strip()
        return {"generation": answer, "final_action": "clarify", "stop_reason": "needs_input"}, {"action": "clarify"}

    def abstain(state, config):
        if state["evidence_status"] == "error":
            answer = "本次检索或证据核对未能完成，暂时无法给出可靠答案，请稍后重试。"
        else:
            answer = "根据当前检索到的资料，无法充分回答该问题。"
            if state.get("missing_points"):
                answer += "仍缺少的信息：" + "；".join(state["missing_points"]) + "。"
        # 没有生成事实答案时，documents 清空表示实际最终回答未使用知识上下文。
        return {"generation": answer, "final_action": "abstain", "documents": []}, {
                    "action": "abstain", "reason": state.get("stop_reason")}

    workflow = StateGraph(GraphState)
    for name, func in [("route", route_question), ("retrieve", retrieve), ("assess", assess),
                       ("rewrite", rewrite), ("generate", generate), ("direct_answer", direct_answer),
                       ("clarify", clarify), ("abstain", abstain)]:
        workflow.add_node(name, measured(name, func))
    workflow.set_entry_point("route")
    workflow.add_conditional_edges("route", decide_retrieve,
                                   {s: s for s in ("retrieve", "direct_answer", "clarify")})
    workflow.add_edge("retrieve", "assess")
    workflow.add_conditional_edges("assess", lambda state: decide_after_grade(state, max_retries),
                                   {s: s for s in ("generate", "rewrite", "clarify", "abstain")})
    workflow.add_edge("rewrite", "retrieve")
    for name in ("generate", "direct_answer", "clarify", "abstain"):
        workflow.add_edge(name, END)
    return workflow.compile()
