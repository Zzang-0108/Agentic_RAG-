"""从逐页 PDF 文本恢复章节；保留每段原文的页号及字符范围。"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from langchain_core.documents import Document
from .provenance import stable_id, validate_provenance


def _lines(page):
    offset, blank = 0, False
    result = []
    for raw in page.page_content.splitlines(keepends=True):
        indent = len(raw) - len(raw.lstrip(" \t"))
        text = raw.rstrip() if indent >= 4 or raw.startswith("\t") else raw.strip()
        if text.strip():
            start = offset if indent >= 4 or raw.startswith("\t") else offset + len(raw) - len(raw.lstrip())
            result.append({"text": text, "start": start, "end": start + len(text), "blank": blank})
        blank = not text
        offset += len(raw)
    return result


def _key(text):
    return re.sub(r"\s+", "", text)


def heading(text):
    """仅识别保守的编号标题；普通短句、公式或年份不当作章节。"""
    if len(text) > 90 or re.search(r"[。！？；;=]", text):
        return None
    if re.match(r"^第\s*[零一二三四五六七八九十百\d]+\s*[章节篇部]", text):
        return (1 if "章" in text or "篇" in text or "部" in text else 2, text)
    if re.match(r"^(?:Chapter|CHAPTER)\s+\d+\b", text):
        return 1, text
    match = re.match(r"^(\d+(?:\.\d+){1,4})\.?\s+([^\d\W].*|[\u4e00-\u9fff].*)$", text)
    if match:
        return match[1].count(".") + 1, text
    return None


def read_outlines(pages, base_dir):
    """PDF 自带书签优先；没有书签时仍可用文本标题规则。"""
    import pymupdf
    result = defaultdict(list)
    base = Path(base_dir).resolve()
    for source in sorted({p.metadata["source"] for p in pages}):
        path = (base / source).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            continue
        with pymupdf.open(path) as pdf:
            for level, title, number in pdf.get_toc(simple=True):
                if number > 0 and title.strip():
                    result[(source, number)].append((level, title.strip()))
    return result


def _looks_like_code_or_math(text):
    return bool(re.match(r"(?:```|~~~|\$|\\\[|\\\]|def |class |import |from .+ import |for .+:|if .+:|>>>|#|[{}])", text)
                or (not re.search(r"[\u4e00-\u9fff]", text)
                    and re.search(r"[=∑∫∏√≤≥]|\\(?:frac|sum|alpha|beta)", text)))


def _separator(previous, current, blank, protected=False):
    if blank:
        return "\n\n"
    if protected or _looks_like_code_or_math(previous) or _looks_like_code_or_math(current):
        return "\n"
    if previous and current and re.search(r"[\u4e00-\u9fff，、]$", previous) and re.match(r"[\u4e00-\u9fff]", current):
        return ""  # 中文排版换行不等于段落边界。
    if re.search(r"[A-Za-z,]$", previous) and re.match(r"[A-Za-z]", current):
        return " "
    return "\n"


def build_sections(pages, outlines=None):
    outlines = outlines or {}
    books = defaultdict(list)
    for page in pages:
        books[page.metadata["source"]].append(page)
    sections, stats = [], Counter()
    for source, book in sorted(books.items()):
        book.sort(key=lambda p: p.metadata["page_number"])
        if len({p.metadata["page_number"] for p in book}) != len(book):
            raise ValueError(f"同一书籍出现重复 PDF 页号：{source}")
        parsed = [_lines(page) for page in book]
        edge_counts = Counter()
        for lines in parsed:
            edge_counts.update({_key(line["text"]) for line in lines[:2] + lines[-2:]
                                if len(line["text"]) <= 60 and not re.search(r"[。！？；;]", line["text"])})
        threshold = min(10, max(3, math.ceil(len(book) * .02)))
        repeated = {key for key, n in edge_counts.items() if n >= threshold}
        seen_headers, title_path = set(), []
        content, spans, size = [], [], 0
        previous_page, previous_line, fenced, previous_heading, has_body = None, "", False, False, False

        def flush():
            nonlocal content, spans, size, previous_line, fenced, previous_heading, has_body
            text = "".join(content)
            if text.strip() and spans:
                title = " / ".join(t for _, t in title_path)
                metadata = {"source": source, "book_name": book[0].metadata.get("book_name", ""),
                            "section_title": title, "section_path": [t for _, t in title_path],
                            "source_spans": spans, "page_numbers": sorted({s["page_number"] for s in spans})}
                metadata["page_number"] = metadata["page_numbers"][0]
                metadata["section_id"] = stable_id("section", source, text, spans)
                doc = Document(page_content=text, metadata=metadata)
                validate_provenance(doc)
                sections.append(doc)
            content, spans, size, previous_line, fenced, previous_heading, has_body = [], [], 0, "", False, False, False

        for page, lines in zip(book, parsed):
            number = page.metadata["page_number"]
            if previous_page is not None and number != previous_page + 1:
                flush()  # 被跳过的空白/扫描页不能假定为连续证据。
            previous_page = number
            # 只剔除明显的目录页；单独的“目录”标题不触发整页删除。
            leaders = sum(bool(re.search(r"(?:[.．·…]\s*){3,}(?:\d+\s*)?$", l["text"])) for l in lines)
            if leaders >= 3:
                flush()
                stats["toc_pages_removed"] += 1
                continue
            cleaned = []
            for index, line in enumerate(lines):
                text = line["text"]
                edge = index < 2 or index >= len(lines) - 2
                adjacent_math = any(_looks_like_code_or_math(lines[j]["text"])
                                    for j in (index - 1, index + 1) if 0 <= j < len(lines))
                page_marker = re.fullmatch(r"\d{1,5}|[ivxlcdm]{2,8}|[IVXLCDM]{2,8}", text)
                if (edge and page_marker and not adjacent_math) or text in {"←_←", "→_→"}:
                    stats["margin_lines_removed"] += 1
                    continue
                key = _key(text)
                if edge and key in repeated and not _looks_like_code_or_math(text):
                    known_heading = (heading(text) or re.fullmatch(r"\d+(?:\.\d+){1,4}\.?", text)
                                     or any(_key(t) == key for _, t in outlines.get((source, number), [])))
                    if not known_heading or key in seen_headers:
                        stats["repeated_headers_removed"] += 1
                        continue
                    seen_headers.add(key)
                cleaned.append(line)
            if not cleaned:
                flush()
                continue
            page_outline = outlines.get((source, number), [])
            title_lines = set()
            for index, line in enumerate(cleaned):
                text = line["text"]
                found = None if fenced or index in title_lines else heading(text)
                # 有些 PDF 把编号和标题抽成两行，例如“2.2\n评估方法”。
                numbered = re.fullmatch(r"\d+(?:\.\d+){1,4}\.?", text)
                chapter_number = re.fullmatch(r"第\s*[零一二三四五六七八九十百\d]+\s*[章节篇部]", text)
                if not fenced and index not in title_lines and (numbered or chapter_number) and index + 1 < len(cleaned):
                    following = cleaned[index + 1]["text"]
                    # 抽出的表格会出现“3.0\nPave\n1\n2.0\nNaN”；邻近多项标量时不猜章节。
                    scalar_rows = sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?|NaN|nan|None|null", item["text"]))
                                      for item in cleaned[index + 2:index + 6])
                    bookmark_match = any(_key(t) in {_key(following), _key(text + following)}
                                         for _, t in page_outline)
                    if (len(following) <= 50 and not re.search(r"[。！？；;=]", following)
                            and re.search(r"[\u4e00-\u9fff]{2}|[A-Za-z]{3}", following)
                            and not _looks_like_code_or_math(following)
                            and (bookmark_match or (following not in {"NaN", "nan", "None", "null"} and scalar_rows < 2))):
                        found = (text.rstrip(".").count(".") + 1 if numbered else found[0], text + " " + following)
                        title_lines.add(index + 1)
                for level, title in page_outline:
                    compact = re.sub(r"\s+", "", text)
                    expected = re.sub(r"\s+", "", title)
                    if not fenced and index not in title_lines and expected == compact:
                        found = level, title
                        break
                if found:
                    # 分行标题的第二行与书签重复匹配时，不再切开编号和标题。
                    if not title_path or found[1] != title_path[-1][1]:
                        if has_body:
                            flush()
                        level, title = found
                        title_path = [(n, t) for n, t in title_path if n < level] + [(level, title)]
                elif index == 0 and page_outline and not any(heading(l["text"]) for l in cleaned):
                    # 无法定位到准确标题时，不凭页号把整页强切成某个书签章节。
                    stats["outline_pages_without_located_heading"] += 1
                is_heading = found is not None or index in title_lines
                separator = ("\n\n" if previous_heading or is_heading else
                             _separator(previous_line, text, line["blank"], fenced)) if content else ""
                content.extend([separator, text])
                size += len(separator)
                spans.append({"start": size, "end": size + len(text), "page_number": number,
                              "page_start": line["start"], "page_end": line["end"]})
                size += len(text)
                if text.startswith(("```", "~~~")):
                    fenced = not fenced
                previous_line = text
                previous_heading = is_heading
                has_body = has_body or not is_heading
        flush()
    stats["sections"] = len(sections)
    stats["cross_page_sections"] = sum(len(d.metadata["page_numbers"]) > 1 for d in sections)
    return sections, dict(stats)
