"""建库工具，与问答/效果评测分开。--preview-structure 完全不调用在线模型。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main(argv=None):
    from .config import BASE_DIR, DOC_DIR, CHUNKING_CONFIG, OPTIMIZED_CACHE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-structure", action="store_true", help="只检查真实语料的结构父子块，不计算语义向量、不建检索索引")
    parser.add_argument("--strategy", choices=["legacy", "structural", "semantic"], default=CHUNKING_CONFIG["strategy"])
    parser.add_argument("--output", type=Path, help="结构预览的新目录；完整索引按配置自动分目录")
    args = parser.parse_args(argv)
    if args.output and not args.preview_structure:
        parser.error("--output 仅用于结构预览；完整构库自动使用独立缓存目录")
    config = {**CHUNKING_CONFIG, "strategy": args.strategy}
    if args.preview_structure:
        from file_loader import load_pdf_documents
        from .chunking import build_chunks
        from .structure import read_outlines
        from .token_budget import TokenCounter
        from .store import write_splits, input_fingerprint
        # 此分支不创建 embedding/LLM；结构预览不冒充已经执行过语义切分。
        config["strategy"] = "structural"
        counter = TokenCounter.load(config["tokenizer"], config.get("tokenizer_revision"))
        pages = load_pdf_documents(DOC_DIR, BASE_DIR)
        chunks = build_chunks(pages, None, counter, config, read_outlines(pages, BASE_DIR))
        raw = {(d.metadata["source"], d.metadata["page_number"]): d.page_content for d in pages}
        for doc in [*chunks.parents.values(), *chunks.children]:
            for span in doc.metadata["source_spans"]:
                if (doc.page_content[span["start"]:span["end"]] !=
                        raw[(doc.metadata["source"], span["page_number"])][span["page_start"]:span["page_end"]]):
                    raise ValueError("来源定位核对失败，不保存预览")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output = args.output or OPTIMIZED_CACHE_DIR / ("preview_structure_" + stamp)
        output.mkdir(parents=True, exist_ok=False)
        write_splits(output / "parents.jsonl", chunks.parents.values())
        write_splits(output / "children.jsonl", chunks.children)
        summary = {"mode": "structure_only_preview", "semantic_embeddings_computed": False,
                   "retrieval_index_built": False, "config": config, "tokenizer_signature": counter.signature,
                   "input_fingerprint": input_fingerprint(DOC_DIR, {**config, "tokenizer_signature": counter.signature}),
                   "stats": chunks.stats, "source_spans_verified": True}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps(summary["stats"], ensure_ascii=False))
        print(f"结构预览已保存：{output}（未运行语义向量计算或问答评测）")
    else:
        from .indexing import build_index
        from .models import make_embeddings
        bundle = build_index(make_embeddings(), chunking_config=config)
        print(f"索引就绪：{bundle.chunk_count} 个检索子块，{len(bundle.parents)} 个父块。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
