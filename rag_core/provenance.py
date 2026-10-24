"""跨页来源映射。offset 是抽取出的 PDF 页文本/当前片段中的字符位置。"""
from __future__ import annotations

import hashlib
import json
from langchain_core.documents import Document


def page_numbers(metadata):
    spans = metadata.get("source_spans")
    values = ([s["page_number"] for s in spans] if spans else
              metadata.get("page_numbers") or [metadata.get("page_number")])
    return sorted({p for p in values if type(p) is int and p > 0})


def stable_id(kind, source, text, spans):
    payload = json.dumps([kind, source, text, spans], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf8")).hexdigest()[:24]


def slice_document(doc, start, end, **extra):
    """截取正文并同步裁剪映射，绝不把父块的所有页号直接复制给子块。"""
    while start < end and doc.page_content[start].isspace():
        start += 1
    while end > start and doc.page_content[end - 1].isspace():
        end -= 1
    spans = []
    original = doc.metadata.get("source_spans") or [{
        "start": 0, "end": len(doc.page_content), "page_number": doc.metadata["page_number"],
        "page_start": 0, "page_end": len(doc.page_content)}]
    for span in original:
        left, right = max(start, span["start"]), min(end, span["end"])
        if left < right:
            page_start = span["page_start"] + left - span["start"]
            spans.append({"start": left - start, "end": right - start,
                          "page_number": span["page_number"], "page_start": page_start,
                          "page_end": page_start + right - left})
    metadata = {**doc.metadata, **extra, "source_spans": spans}
    pages = page_numbers({"source_spans": spans})
    metadata.update(page_numbers=pages, page_number=pages[0] if pages else None)
    return Document(page_content=doc.page_content[start:end], metadata=metadata)


def validate_provenance(doc):
    previous = 0
    for span in doc.metadata.get("source_spans", []):
        if not (previous <= span["start"] < span["end"] <= len(doc.page_content)):
            raise ValueError("来源映射发生重叠或越界")
        if span["end"] - span["start"] != span["page_end"] - span["page_start"]:
            raise ValueError("片段与 PDF 页的来源长度不一致")
        if span["page_start"] < 0 or span["page_number"] < 1:
            raise ValueError("非法 PDF 页来源")
        previous = span["end"]

