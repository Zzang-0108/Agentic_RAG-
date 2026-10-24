"""请求独立的节点计时、调用量及候选定位信息，不使用共享 last_trace。"""
from __future__ import annotations
import hashlib
from langchain_core.callbacks import BaseCallbackHandler
from .provenance import page_numbers


def document_id(doc):
    if doc.metadata.get("chunk_id"):
        return doc.metadata["chunk_id"]
    value = "\0".join((str(doc.metadata.get("source", "")),
                        str(doc.metadata.get("page_number", "")), doc.page_content))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def dedupe_documents(documents):
    """按片段标识去重，保留首次出现的文档及原有顺序。"""
    seen = set()
    result = []
    for doc in documents:
        key = document_id(doc)
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def document_snapshot(doc, rank):
    return {"chunk_id": document_id(doc), "rank": rank,
            "source": doc.metadata.get("source", ""), "page_number": doc.metadata.get("page_number"),
            "page_numbers": page_numbers(doc.metadata), "source_spans": doc.metadata.get("source_spans", []),
            "parent_id": doc.metadata.get("parent_id"), "chunk_type": doc.metadata.get("chunk_type"),
            "section_title": doc.metadata.get("section_title"), "token_count": doc.metadata.get("token_count"),
            "matched_children": doc.metadata.get("matched_children", []),
            "expansion": doc.metadata.get("expansion"),
            "rerank_score": doc.metadata.get("rerank_score"),
            "retrieval_score": doc.metadata.get("score"), "text": doc.page_content}


class NodeUsage(BaseCallbackHandler):
    def __init__(self):
        self.started, self.finished, self.failed = set(), set(), set()
        self.tokens = self.reported = 0

    def on_chat_model_start(self, *args, run_id=None, **kwargs):
        self.started.add(str(run_id))

    def on_llm_start(self, *args, run_id=None, **kwargs):
        self.started.add(str(run_id))

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        self.failed.add(str(run_id))

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        key = str(run_id)
        if key in self.finished:
            return
        self.finished.add(key)
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        total = usage.get("total_tokens")
        if total is None:
            for batch in getattr(response, "generations", []):
                for generation in batch:
                    message = getattr(generation, "message", None)
                    value = (getattr(message, "usage_metadata", None) or {}).get("total_tokens")
                    if value is not None:
                        total = (total or 0) + value
        if total is not None:
            self.tokens += int(total)
            self.reported += 1

    def snapshot(self):
        return {"llm_started_calls": len(self.started), "llm_failed_calls": len(self.failed),
                "token_reported_calls": self.reported,
                "reported_total_tokens": self.tokens if self.reported else None}
