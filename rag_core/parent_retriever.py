"""先重排子块，再补回父块；来源与子块得分分别记录。"""
from __future__ import annotations
from langchain_core.retrievers import BaseRetriever
from .provenance import slice_document
from .telemetry import document_id, document_snapshot


def child_fallbacks(parent):
    for match in parent.metadata.get("matched_children", []):
        doc = slice_document(parent, match["parent_start"], match["parent_end"],
                             chunk_type="child", chunk_id=match["chunk_id"],
                             expansion="child_budget_fallback", matched_children=[])
        doc.metadata.update(token_count=match["token_count"], rerank_score=match.get("rerank_score"),
                            score=match.get("retrieval_score"), parent_start=match["parent_start"],
                            parent_end=match["parent_end"])
        yield doc


def pack_context(documents, measure, budget):
    """measure 计量整个列表。父块放不下时优先保留已命中的子块，不截父块开头。"""
    selected, seen, omitted, fallbacks = [], set(), [], 0
    for doc in documents:
        key = document_id(doc)
        if key in seen:
            continue
        if measure(selected + [doc]) <= budget:
            selected.append(doc)
            seen.add(key)
            continue
        added = False
        for child in child_fallbacks(doc):
            key = document_id(child)
            if key not in seen and measure(selected + [child]) <= budget:
                selected.append(child)
                seen.add(key)
                added = True
                fallbacks += 1
        if not added:
            omitted.append(document_id(doc))
    return selected, {"budget_tokens": budget, "used_tokens": measure(selected),
                      "child_fallbacks": fallbacks, "omitted_chunk_ids": omitted}


class ParentContextRetriever(BaseRetriever):
    base_retriever: object
    parents: dict
    token_counter: object
    max_context_tokens: int = 4096

    @property
    def k(self):
        return getattr(self.base_retriever, "k", None)

    def _get_relevant_documents(self, query, *, run_manager=None):
        config = {"callbacks": run_manager.get_child()} if run_manager else None
        return self.retrieve_with_trace(query, config)[0]

    def retrieve_with_trace(self, query, config=None):
        if hasattr(self.base_retriever, "retrieve_with_trace"):
            children, trace = self.base_retriever.retrieve_with_trace(query, config=config)
        else:
            children = self.base_retriever.invoke(query, config=config)
            trace = {"selected": [document_snapshot(d, i) for i, d in enumerate(children, 1)]}
        grouped = {}
        for child in children:
            parent_id = child.metadata.get("parent_id")
            parent = self.parents.get(parent_id)
            if parent is None:
                raise ValueError(f"子块缺少对应父块：{parent_id}，请检查优化索引完整性")
            start, end = child.metadata["parent_start"], child.metadata["parent_end"]
            if parent.metadata["source"] != child.metadata["source"] or parent.page_content[start:end] != child.page_content:
                raise ValueError("父子块来源或字符范围不匹配")
            grouped.setdefault(parent_id, []).append(child)
        expanded = []
        for parent_id, matches in grouped.items():
            parent = self.parents[parent_id]
            detail = [{"chunk_id": d.metadata["chunk_id"], "parent_start": d.metadata["parent_start"],
                       "parent_end": d.metadata["parent_end"], "token_count": d.metadata["token_count"],
                       "rerank_score": d.metadata.get("rerank_score"), "retrieval_score": d.metadata.get("score")}
                      for d in matches]
            expanded.append(parent.model_copy(update={"metadata": {
                **parent.metadata, "matched_children": detail, "expansion": "parent"}}))
        selected, budget = pack_context(expanded, lambda docs: sum(self.token_counter.count(d.page_content) for d in docs),
                                        self.max_context_tokens)
        return selected, {**trace, "parent_expansion": {
            **budget, "tokenizer": self.token_counter.model_name,
            "matched_child_count": len(children), "unique_parent_count": len(grouped),
            "selected": [document_snapshot(d, i) for i, d in enumerate(selected, 1)]}}
