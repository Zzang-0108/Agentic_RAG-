"""结构父块 + 中文句段/语义子块。语义判断只在建库时发生。"""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
from .provenance import slice_document, stable_id
from .structure import build_sections
from .token_budget import bounded_end


@dataclass
class ChunkBundle:
    children: list
    parents: dict
    stats: dict


PROTECTED = re.compile(r"^```[^\n]*\n.*?(?:^```[^\n]*$|\Z)|^~~~[^\n]*\n.*?(?:^~~~[^\n]*$|\Z)|\$\$.*?\$\$|\\\[.*?\\\]", re.M | re.S)
SENTENCE_END = re.compile(r'[。！？]+[”’」』"]*|(?<!\d)[.!?]+(?=\s|$)|\n[ \t]*\n')


def sentence_units(text, count, max_tokens, max_chars=None):
    """返回原文字符区间。保留代码/显式公式块，过长原子块才强制切开并标记。"""
    raw, cursor = [], 0
    def prose(start, end):
        for match in SENTENCE_END.finditer(text, start, end):
            raw.append((start, match.end()))
            start = match.end()
        if start < end:
            raw.append((start, end))
    for match in PROTECTED.finditer(text):
        prose(cursor, match.start())
        raw.append((match.start(), match.end()))
        cursor = match.end()
    prose(cursor, len(text))
    units = []
    for start, end in raw:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        forced = count(text[start:end]) > max_tokens or (max_chars is not None and end - start > max_chars)
        while start < end:
            right = bounded_end(text, start, end, count, max_tokens, max_chars)
            if text[start:right].strip():
                units.append((start, right, bool(forced)))
            start = right
    return units


def group_units(text, units, count, max_tokens, max_chars=None, break_after=(), min_tokens=0):
    groups, first = [], 0
    for i in range(1, len(units)):
        start, end = units[first][0], units[i][1]
        overflow = count(text[start:end]) > max_tokens or (max_chars is not None and end - start > max_chars)
        semantic_break = i - 1 in break_after and count(text[start:units[i - 1][1]]) >= min_tokens
        if overflow or semantic_break:
            groups.append((first, i))
            first = i
    if units:
        groups.append((first, len(units)))
    # 小尾块可以合并时才合并；绝不丢弃短公式或使最后一块超过硬上限。
    if len(groups) > 1:
        left, right = groups[-1]
        previous = groups[-2][0]
        start, end = units[previous][0], units[right - 1][1]
        if (count(text[units[left][0]:end]) < min_tokens
                and count(text[start:end]) <= max_tokens
                and (max_chars is None or end - start <= max_chars)):
            groups[-2:] = [(previous, right)]
    return groups


def semantic_breaks(text, units, embeddings, count, config):
    if len(units) < 2:
        return set()
    windows = []
    radius = config["semantic_buffer"]
    for i, (start, end, _) in enumerate(units):
        left, right = start, end
        for distance in range(1, radius + 1):
            for a, b in ((max(0, i - distance), i), (i, min(len(units) - 1, i + distance))):
                new_left, new_right = min(left, units[a][0]), max(right, units[b][1])
                candidate = text[new_left:new_right]
                if len(candidate) <= config["embedding_max_chars"] and count(candidate) <= config["child_max_tokens"]:
                    left, right = new_left, new_right
        windows.append(text[left:right])
    vectors = np.asarray(embeddings.embed_documents(windows), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(windows) or not np.isfinite(vectors).all():
        raise ValueError("语义切分的 embedding 数量、维度或数值无效")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("语义切分不能使用零向量")
    normalized = vectors / norms
    distances = 1 - np.clip(np.sum(normalized[:-1] * normalized[1:], axis=1), -1, 1)
    threshold = float(np.percentile(distances, config["semantic_percentile"]))
    return {i for i, distance in enumerate(distances) if distance > max(threshold, 1e-6)}


def validate_config(config):
    if config["strategy"] not in {"structural", "semantic"}:
        raise ValueError("优化切分 strategy 必须为 structural 或 semantic")
    if not 0 <= config["semantic_percentile"] <= 100 or config["semantic_buffer"] < 0:
        raise ValueError("语义阈值应在 0~100，邻句窗口应非负")
    if not 0 <= config["child_min_tokens"] <= config["child_max_tokens"] <= config["parent_max_tokens"]:
        raise ValueError("需要 0 <= child_min_tokens <= child_max_tokens <= parent_max_tokens")
    if config["child_max_tokens"] < 1 or config["embedding_max_chars"] < 1:
        raise ValueError("片段长度上限必须为正")


def build_chunks(pages, embeddings, counter, config, outlines=None):
    validate_config(config)
    if config["strategy"] == "semantic" and embeddings is None:
        raise ValueError("semantic 策略需要 embedding 实例，不会静默降级为字符切分")
    count = counter.count
    sections, stats = build_sections(pages, outlines)
    print(f"开始 {config['strategy']} 切分：{len(sections)} 个章节段 …", flush=True)
    parents, children = {}, []
    semantic_boundaries, forced_children = 0, 0
    for section in sections:
        parent_units = sentence_units(section.page_content, count, config["parent_max_tokens"])
        groups = group_units(section.page_content, parent_units, count, config["parent_max_tokens"])
        for a, b in groups:
            parent = slice_document(section, parent_units[a][0], parent_units[b - 1][1], chunk_type="parent")
            parent_id = stable_id("parent", parent.metadata["source"], parent.page_content, parent.metadata["source_spans"])
            parent.metadata.update(parent_id=parent_id, chunk_id=parent_id, token_count=count(parent.page_content),
                                   forced_split=any(u[2] for u in parent_units[a:b]))
            parents[parent_id] = parent
            units = sentence_units(parent.page_content, count, config["child_max_tokens"], config["embedding_max_chars"])
            breaks = (semantic_breaks(parent.page_content, units, embeddings, count, config)
                      if config["strategy"] == "semantic" else set())
            semantic_boundaries += len(breaks)
            child_groups = group_units(parent.page_content, units, count, config["child_max_tokens"],
                                       config["embedding_max_chars"], breaks, config["child_min_tokens"])
            for left, right in child_groups:
                start, end = units[left][0], units[right - 1][1]
                child = slice_document(parent, start, end, chunk_type="child", parent_id=parent_id,
                                       parent_start=start, parent_end=end, strategy=config["strategy"],
                                       forced_split=parent.metadata["forced_split"] or any(u[2] for u in units[left:right]))
                # slice_document 会去掉边缘空白，parent_start/end 必须对应实际正文。
                child.metadata["parent_start"] = start + len(parent.page_content[start:end]) - len(parent.page_content[start:end].lstrip())
                child.metadata["parent_end"] = child.metadata["parent_start"] + len(child.page_content)
                child.metadata["chunk_id"] = stable_id("child", child.metadata["source"], child.page_content, child.metadata["source_spans"])
                child.metadata["token_count"] = count(child.page_content)
                forced_children += int(child.metadata["forced_split"])
                children.append(child)
            if len(parents) % 100 == 0:
                print(f"切分进度：已处理 {len(parents)} 个父块，生成 {len(children)} 个子块", flush=True)
    stats.update(parents=len(parents), children=len(children), semantic_boundaries=semantic_boundaries,
                 forced_split_children=forced_children,
                 cross_page_children=sum(len(d.metadata["page_numbers"]) > 1 for d in children),
                 cross_page_parents=sum(len(d.metadata["page_numbers"]) > 1 for d in parents.values()))
    return ChunkBundle(children, parents, stats)
