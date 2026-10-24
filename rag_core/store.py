"""索引落盘：把建索引的成果存到磁盘，下次启动直接读回，不再解析 PDF、不再分词。

当前索引在 `embed_cache_optimized/` 下按切分策略与模型配置使用独立子目录：

    meta.json      指纹与校验信息（最后写、写完整才把 complete 置 true）
    splits.jsonl   切分后的文本片段（正文 + metadata）
    parents.jsonl  优化策略的父块（不参与向量检索；legacy 无此文件）
    vectors.npy    embedding 原始输出（本地 BGE 已归一化；旧接口输出按原样保存）
    hnsw.index     faiss HNSW 索引
    bm25.pkl       BM25Retriever（pickle）

为什么要单独算一个「输入指纹」：
    重构前判断缓存是否命中，得先把 4 本 PDF 解析、切分、算出 corpus_hash 才知道，
    光解析就要 5.6 秒，缓存省不下什么。现在改成只看输入——
    PDF 本身的 sha256 + 切分参数 + embedding 模型 + schema 版本——
    算一次 0.63 秒，不用解析 PDF 就能判断命中。
    原来的 corpus_hash 继续照算、照存，它负责校验向量缓存（保证一次 embedding 接口都不白调），
    同时也是重构前后的回归基准。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from langchain_core.documents import Document

from .config import BASE_DIR, EMBEDDING_MODEL, EMBEDDING_CONFIG, SCHEMA_VERSION, SPLIT_CONFIG

META_NAME = "meta.json"
SPLITS_NAME = "splits.jsonl"
VECTORS_NAME = "vectors.npy"
HNSW_NAME = "hnsw.index"
BM25_NAME = "bm25.pkl"
PARENTS_NAME = "parents.jsonl"

# 参与指纹计算的加载器源码：防止「改了加载逻辑，却忘了把 SCHEMA_VERSION +1」导致静默用错缓存
LOADER_FILE = BASE_DIR / "file_loader.py"


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    """分块计算文件 sha256（1MB 一块，不把 256MB 的 PDF 整个读进内存）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """尽量转成相对项目根的路径，转不了就用绝对路径。"""
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def input_fingerprint(doc_dir: str | Path, chunking_config=None) -> str:
    """只根据输入算指纹，不解析 PDF。"""
    doc_dir = Path(doc_dir)
    files = [
        [_relative(pdf), pdf.stat().st_size, sha256_file(pdf)]
        for pdf in sorted(
            p for p in doc_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
        )
    ]
    payload = {
        "files": files,
        "loader_sha256": sha256_file(LOADER_FILE) if LOADER_FILE.is_file() else None,
        "split_config": SPLIT_CONFIG,
        "embed_model": EMBEDDING_MODEL,
        "embedding_config": EMBEDDING_CONFIG,
        "schema_version": SCHEMA_VERSION,
    }
    if chunking_config is not None:
        payload["chunking_config"] = chunking_config
        payload["chunking_source_hashes"] = {
            name: sha256_file(Path(__file__).parent / name)
            for name in ("chunking.py", "structure.py", "provenance.py", "token_budget.py")}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _replace(tmp: Path, final: Path) -> None:
    """原子替换：先写 .tmp，再 os.replace（同一磁盘上是原子操作）。

    中途崩溃最多留下一个 .tmp 垃圾文件，不会把原来好好的缓存写成半截。
    """
    os.replace(tmp, final)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace(tmp, path)


def write_meta(cache_dir: str | Path, meta: dict) -> None:
    _write_text_atomic(
        Path(cache_dir) / META_NAME,
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_splits(path: str | Path, splits: Iterable[Document]) -> None:
    """一行一个片段。default=str 兜底：PyMuPDF 给的 metadata 里可能有非 JSON 类型。"""
    lines = [
        json.dumps(
            {"page_content": doc.page_content, "metadata": doc.metadata},
            ensure_ascii=False,
            default=str,
        )
        for doc in splits
    ]
    _write_text_atomic(Path(path), "\n".join(lines) + "\n")


def read_splits(path: str | Path) -> list[Document]:
    documents = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        documents.append(
            Document(page_content=item["page_content"], metadata=item["metadata"])
        )
    return documents


def save_bm25(cache_dir: str | Path, bm25_retriever) -> None:
    """单独重写 bm25.pkl（反序列化失败时用它原地修复，不用整体重建）。"""
    bm25_tmp = Path(cache_dir) / (BM25_NAME + ".tmp")
    with bm25_tmp.open("wb") as stream:
        pickle.dump(bm25_retriever, stream)
    _replace(bm25_tmp, Path(cache_dir) / BM25_NAME)


def save(
    cache_dir: str | Path,
    fingerprint: str,
    cache_key: dict,
    splits: list[Document],
    vectors: np.ndarray,
    hnsw_index,
    bm25_retriever,
    parents=None,
) -> None:
    """落盘。meta.json 写两次：先标记 complete=false，全部写完再置 true。

    这样即使在第 3 个文件处断电，下次启动也会看到 complete=false 而重新构建，
    不会把「一半旧的、一半新的」缓存当成好的用。

    注意：必须在**任何检索发生之前**调用。HNSWRetriever 会往共用的
    doc.metadata["score"] 里写分数，查过之后再落盘，splits.jsonl 就不是
    「纯语料」了（不影响 corpus_hash，因为那个只看 source/页码/正文，
    但文件会变得不可复现）。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(cache_key)
    meta["input_fingerprint"] = fingerprint
    meta["complete"] = False
    write_meta(cache_dir, meta)

    write_splits(cache_dir / SPLITS_NAME, splits)
    if parents is not None:
        validate_parent_links(splits, parents)
        write_splits(cache_dir / PARENTS_NAME, parents.values())
        meta["parent_count"] = len(parents)
        meta["parents_sha256"] = sha256_file(cache_dir / PARENTS_NAME)
        meta["children_sha256"] = sha256_file(cache_dir / SPLITS_NAME)

    # 传文件对象给 np.save，它才不会自作主张再补一个 .npy 后缀
    vectors_tmp = cache_dir / (VECTORS_NAME + ".tmp")
    with vectors_tmp.open("wb") as stream:
        np.save(stream, vectors)
    _replace(vectors_tmp, cache_dir / VECTORS_NAME)

    hnsw_tmp = cache_dir / (HNSW_NAME + ".tmp")
    faiss.write_index(hnsw_index, str(hnsw_tmp))
    _replace(hnsw_tmp, cache_dir / HNSW_NAME)

    # 注意：pickle 里存的是对 rag_core.indexing.jieba_cut 这个函数全名的引用，
    # 那个函数改名或搬家会让旧缓存反序列化失败（删掉 bm25.pkl 重建即可，代价 6 秒）。
    bm25_tmp = cache_dir / (BM25_NAME + ".tmp")
    with bm25_tmp.open("wb") as stream:
        pickle.dump(bm25_retriever, stream)
    _replace(bm25_tmp, cache_dir / BM25_NAME)

    meta["complete"] = True
    write_meta(cache_dir, meta)


def load(cache_dir: str | Path, fingerprint: str) -> dict | None:
    """命中返回 {meta, splits, vectors, hnsw_index, bm25_retriever}，任何一项不对就返回 None。

    唯一的例外：bm25.pkl 反序列化失败时不算「整包作废」，而是返回
    bm25_retriever=None，让调用方现场重建这一路（向量和 HNSW 都还能用）。
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / META_NAME
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("meta.json 读不了或不是合法 JSON，重新建立索引")
        return None
    if meta.get("complete") is not True:
        print("上次落盘没写完（complete 不是 true），重新建立索引")
        return None
    if meta.get("input_fingerprint") != fingerprint:
        print("缓存指纹变了（换过 PDF / 切分参数 / embedding 模型 / 加载器），重新建立索引")
        return None

    missing = [
        name
        for name in (SPLITS_NAME, VECTORS_NAME, HNSW_NAME, BM25_NAME)
        if not (cache_dir / name).is_file()
    ]
    if missing:
        print(f"落盘文件缺失：{missing}，重新建立索引")
        return None

    if "chunking_config" in meta and "parent_count" not in meta:
        print("优化索引缺少父块清单，重新构建")
        return None
    if "children_sha256" in meta and sha256_file(cache_dir / SPLITS_NAME) != meta["children_sha256"]:
        print("子块文件校验失败，重新构建优化索引")
        return None
    splits = read_splits(cache_dir / SPLITS_NAME)
    parents = {}
    if "parent_count" in meta:
        path = cache_dir / PARENTS_NAME
        if not path.is_file() or sha256_file(path) != meta.get("parents_sha256"):
            print("父块文件缺失或校验失败，重新建立优化索引")
            return None
        try:
            parent_docs = read_splits(path)
            parents = {d.metadata["parent_id"]: d for d in parent_docs}
            if len(parent_docs) != len(parents) or len(parents) != meta["parent_count"]:
                raise ValueError("父块数量不一致")
            validate_parent_links(splits, parents)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"父子块映射无效：{exc}，重新建立优化索引")
            return None
    vectors = np.load(cache_dir / VECTORS_NAME)
    if vectors.ndim != 2:
        print("vectors.npy 不是二维数组，重新建立索引")
        return None
    if not np.isfinite(vectors).all():
        # 与重构前 rag.py 里的检查一致：向量里的 NaN/Inf 一旦进了 faiss 索引，
        # 检索会静默返回垃圾结果，不如在这里就拦下来
        print("vectors.npy 里有 NaN 或无穷值，重新建立索引")
        return None
    hnsw_index = faiss.read_index(str(cache_dir / HNSW_NAME))

    # bm25.pkl 是 pickle，只要 jieba_cut 改过名/搬过家就会反序列化失败。
    # 这种情况只重建 BM25（重新分词约 8 秒、不花钱），
    # 绝不能让它把整个 import 拖挂——向量和 HNSW 都还在，一分钱不用花。
    bm25_retriever = None
    try:
        with (cache_dir / BM25_NAME).open("rb") as stream:
            bm25_retriever = pickle.load(stream)
    except Exception as exc:  # noqa: BLE001 —— pickle 能抛的异常种类太多了
        print(f"bm25.pkl 读不出来（{type(exc).__name__}: {exc}）")
        print("只重建 BM25，向量与 HNSW 继续用落盘的")

    # 完整性校验：四处数量必须一致
    counts = {
        "splits.jsonl": len(splits),
        "vectors.npy": int(vectors.shape[0]),
        "hnsw.index": int(hnsw_index.ntotal),
        "meta.chunk_count": meta.get("chunk_count"),
    }
    if len(set(counts.values())) != 1:
        print(f"落盘缓存数量对不上：{counts}，重新建立索引")
        return None

    return {
        "meta": meta,
        "splits": splits,
        "vectors": vectors,
        "hnsw_index": hnsw_index,
        "bm25_retriever": bm25_retriever,
        "parents": parents,
    }


def validate_parent_links(children, parents):
    from .provenance import validate_provenance, slice_document
    for parent in parents.values():
        validate_provenance(parent)
    for child in children:
        parent = parents.get(child.metadata.get("parent_id"))
        if parent is None:
            raise ValueError("存在无父块的子块")
        start, end = child.metadata["parent_start"], child.metadata["parent_end"]
        if not 0 <= start < end <= len(parent.page_content):
            raise ValueError("父块定位越界")
        if (child.metadata["source"] != parent.metadata["source"]
                or parent.page_content[start:end] != child.page_content):
            raise ValueError("父子块正文/来源不一致")
        validate_provenance(child)
        if slice_document(parent, start, end).metadata["source_spans"] != child.metadata.get("source_spans"):
            raise ValueError("子块来源映射与父块不一致")
