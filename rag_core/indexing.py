"""语料 → 切分 → 向量 → 三路检索器。

对外只需要一个入口：

    bundle = build_index(embeddings)

不传 chunking_config 时保留 legacy 行为；rag.py 显式传入当前优化配置。
各策略按 embedding 编码配置隔离目录；旧智谱索引保留在 embed_cache_books/。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import jieba
import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 注意：file_loader 是仓库根目录下的顶层模块，没有跟着搬进包，所以这里
# 在函数内部延迟导入——这样测试只 import rag_core.indexing 时不需要它，
# 等真正要解析 PDF 的时候再导入（导入失败点仍然是 import rag，行为不变）。
from . import store
from .config import (
    BASE_DIR,
    BATCH_SIZE,
    DOC_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_CONFIG,
    ENSEMBLE_WEIGHTS,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    SCHEMA_VERSION,
    SPLIT_CONFIG,
    TOP_K,
    OPTIMIZED_CACHE_DIR,
)


# ========== 1. 把 faiss 索引封装成 LangChain Retriever ==========
class HNSWRetriever(BaseRetriever):
    """把 faiss.IndexHNSWFlat 包装成 LangChain Retriever"""

    index: object
    documents: List[Document]
    # 注意：这里传进去的是 splits 这个列表本身，pydantic 默认不会再复制一层
    # （实测 self.documents[0] is splits[0] 为 True），所以下面往
    # doc.metadata["score"] 里写，改的是 BM25 / Chroma 共用的同一批 Document 对象，是有副作用的。
    #
    # ⚠️ 上面这句对 BM25 部分不成立，实测：
    #   - HNSWRetriever：列表本身是副本，但里面的 Document 元素和它们的 metadata
    #     就是 splits 里的同一批对象 -> 确实有副作用。
    #   - BM25Retriever.from_documents：内部走 from_texts，会造**全新**的 Document
    #     和**全新**的 metadata dict（langchain_community/retrievers/bm25.py），
    #     所以 score 写不到 BM25 那边的文档上。
    #   结论：Bug 还是那个 Bug（跳过的下标会错位），但"污染 BM25"这层担心不用有。
    #   另外因为 splits 与检索器共享 Document，**检索之后再落盘**会把 score 写进
    #   splits.jsonl，所以 store.save() 只能在任何查询之前调用。
    embeddings: object
    k: int = 5

    def _get_relevant_documents(self, query, *, run_manager=None) -> List[Document]:
        query_vector = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, self.k)
        result = []
        for idx in indices[0]:
            if idx != -1:
                doc = self.documents[idx]
                # 用 len(result) 当下标取分数是错的：一旦上面跳过 idx == -1，
                # len(result) 就和 scores[0] 里的位置对不上，后面的分数会整体错位。
                # 正确写法是连循环下标一起取：
                #     for pos, idx in enumerate(indices[0]):
                #         if idx != -1:
                #             doc.metadata["score"] = float(scores[0][pos])
                doc.metadata["score"] = float(scores[0][len(result)])
                result.append(doc)
        return result


# ========== 2. BM25 的中文分词 ==========
def jieba_cut(text: str) -> list[str]:
    """BM25 的分词函数。

    必须是模块级具名函数，不能写成 lambda：bm25.pkl 里保存的是
    「rag_core.indexing.jieba_cut」这个全名引用，而 lambda 无法 pickle
    （PicklingError: Can't pickle <lambda>）。所以别改它的名字或位置。
    """
    return list(jieba.cut(text))


# ========== 3. 指纹与向量缓存 ==========
def corpus_hash(splits: list[Document]) -> str:
    """根据实际切分结果生成指纹：包含顺序、来源、页码和正文，确保向量与片段对应。"""
    corpus_data = []
    for doc in splits:
        row = [doc.metadata["source"], doc.metadata["page_number"], doc.page_content]
        if "source_spans" in doc.metadata:
            row.append({key: doc.metadata.get(key) for key in
                        ("source_spans", "parent_id", "parent_start", "parent_end", "section_title")})
        corpus_data.append(row)
    digest = hashlib.sha256(
        json.dumps(corpus_data, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    del corpus_data
    return digest


def make_cache_key(corpus_digest: str, chunk_count: int, chunking_config=None) -> dict:
    """记录模型版本与编码方式，避免混用不同向量空间。"""
    key = {
        "schema_version": SCHEMA_VERSION,
        "corpus_hash": corpus_digest,
        "chunk_count": chunk_count,
        "split_config": SPLIT_CONFIG,
        "embed_model": EMBEDDING_MODEL,
        "embedding_config": EMBEDDING_CONFIG,
    }
    if chunking_config is not None:
        key["chunking_config"] = chunking_config
    return key


def encode_documents(embeddings, texts: list[str]) -> np.ndarray:
    """分批调 embedding 接口编码文档。"""
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        all_vectors.extend(embeddings.embed_documents(batch))
        print(f"已编码 {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.array(all_vectors, dtype=np.float32)


def get_vectors(embeddings, splits: list[Document], corpus_digest: str, cache_dir: Path, chunking_config=None) -> np.ndarray:
    """复用已有的向量缓存，或重新编码。

    meta.json 中 cache_key 的全部字段相同才复用。更换模型后不复用旧向量；
    优化策略改变了片段正文，需要单独编码，已完成文本通过 SQLite 缓存复用。
    """
    cache_key = make_cache_key(corpus_digest, len(splits), chunking_config)
    meta_path = cache_dir / store.META_NAME
    vectors_path = cache_dir / store.VECTORS_NAME
    if meta_path.is_file() and vectors_path.is_file():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if all(old.get(key) == value for key, value in cache_key.items()):
            vectors = np.load(vectors_path)
            print(f"命中向量缓存：读入 {vectors.shape[0]} 条向量，跳过 embedding 接口")
            return vectors
        print("向量缓存指纹变了（换过 PDF / 切分参数 / embedding 模型），重新编码")
    print("正在编码文档向量...")
    return encode_documents(embeddings, [doc.page_content for doc in splits])


# ========== 4. 建 HNSW 索引与三路检索器 ==========
def build_faiss_index(vectors: np.ndarray):
    """建 faiss HNSW 索引（向量已归一化，所以内积就等于余弦相似度）。"""
    dim = vectors.shape[1]
    # 每个节点在图中连多少个邻居。这是构造期参数，add() 建完图之后就改不了了。
    # M 越大，图连通性越好（召回越高），但内存和建索引时间都越大。
    # faiss.METRIC_INNER_PRODUCT：用内积作为相似度度量（即"越大越相似"）
    hnsw_index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    # 插入每个新节点时，需要在已有图中找它的 M 个最近邻居；这一步本身也是一次图搜索，
    # efConstruction 就是这次搜索的候选池大小。
    hnsw_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw_index.add(vectors)
    hnsw_index.hnsw.efSearch = HNSW_EF_SEARCH  # 查询时的搜索深度，越大召回越高、越慢，可按需调
    print(f"向量库构建完成，向量维度: {dim}, 节点数: {hnsw_index.ntotal}")
    return hnsw_index


def build_hnsw_retriever(splits: list[Document], hnsw_index, embeddings) -> HNSWRetriever:
    return HNSWRetriever(index=hnsw_index, documents=splits, embeddings=embeddings, k=TOP_K)


def build_bm25_retriever(splits: list[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(splits, k=TOP_K, preprocess_func=jieba_cut)


def build_ensemble_retriever(hnsw_retriever, bm25_retriever) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[hnsw_retriever, bm25_retriever], weights=ENSEMBLE_WEIGHTS,
        id_key=("chunk_id" if all(d.metadata.get("chunk_id") for d in hnsw_retriever.documents) else None),
    )


def _normalize(vectors: np.ndarray) -> None:
    """原地归一化，之后内积就等于余弦相似度。"""
    faiss.normalize_L2(vectors)


def _report(splits: list[Document]) -> None:
    """打印语料规模。缓存命中时也从 splits 反推，两条路径输出一致。"""
    book_count = len({doc.metadata["source"] for doc in splits})
    page_count = len(
        {(doc.metadata["source"], p) for doc in splits
         for p in doc.metadata.get("page_numbers", [doc.metadata.get("page_number")])}
    )
    print(f"书籍数：{book_count}\n有效页面数：{page_count}\n文本片段数：{len(splits)}")


# ========== 5. 入口 ==========
@dataclass
class IndexBundle:
    """一次构建的全部产物。"""

    splits: list[Document]
    vectors: np.ndarray
    hnsw_index: object
    hnsw_retriever: HNSWRetriever
    bm25_retriever: BM25Retriever
    ensemble_retriever: EnsembleRetriever
    cache_key: dict
    fingerprint: str
    from_cache: bool
    parents: dict = field(default_factory=dict)
    token_counter: object | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.splits)


def build_index(
    embeddings,
    doc_dir: str | Path = DOC_DIR,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    chunking_config: dict | None = None,
    token_counter=None,
) -> IndexBundle:
    """准备配置，读取或新建索引，再统一组装检索器与返回结果。"""
    doc_dir = Path(doc_dir)
    if chunking_config is not None and chunking_config.get("strategy") == "legacy":
        chunking_config = None

    # 缓存目录按切分配置和模型标识隔离。
    if chunking_config is not None:
        from .chunking import validate_config
        from .token_budget import TokenCounter

        chunking_config = dict(chunking_config)
        validate_config(chunking_config)
        if not token_counter:
            token_counter = TokenCounter.load(
                chunking_config["tokenizer"], chunking_config.get("tokenizer_revision")
            )
        chunking_config["tokenizer_signature"] = token_counter.signature
        key = hashlib.sha256(
            json.dumps([chunking_config, EMBEDDING_CONFIG], sort_keys=True).encode()
        ).hexdigest()[:12]
        default_cache_dir = OPTIMIZED_CACHE_DIR / (chunking_config["strategy"] + "_" + key)
    else:
        key = hashlib.sha256(json.dumps(EMBEDDING_CONFIG, sort_keys=True).encode()).hexdigest()[:12]
        default_cache_dir = OPTIMIZED_CACHE_DIR / ("legacy_" + key)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir

    fingerprint = store.input_fingerprint(doc_dir, chunking_config)
    cached = None if force_rebuild else store.load(cache_dir, fingerprint)
    from_cache = cached is not None

    if from_cache:
        print(f"命中落盘索引（指纹 {fingerprint[:12]}）：跳过 PDF 解析 / 分词 / 建图")
        splits = cached["splits"]
        vectors = cached["vectors"]
        _normalize(vectors)
        hnsw_index = cached["hnsw_index"]
        hnsw_retriever = build_hnsw_retriever(splits, hnsw_index, embeddings)
        # 查询期参数直接对齐当前配置，不为修改 k 或 efSearch 重建向量库。
        hnsw_index.hnsw.efSearch = HNSW_EF_SEARCH
        bm25_retriever = cached["bm25_retriever"]
        if bm25_retriever is None:
            bm25_retriever = build_bm25_retriever(splits)
            store.save_bm25(cache_dir, bm25_retriever)
            print("bm25.pkl 已就地修好")
        elif bm25_retriever.k != TOP_K:
            print(f"落盘的 BM25 是 k={bm25_retriever.k}，按当前配置对齐到 {TOP_K}")
            bm25_retriever.k = TOP_K

        parents = cached.get("parents", {})
        cache_key = {
            name: cached["meta"][name]
            for name in ("schema_version", "corpus_hash", "chunk_count", "split_config",
                         "embed_model", "embedding_config", "chunking_config", "chunking_stats")
            if name in cached["meta"]
        }
    else:
        from file_loader import load_pdf_documents

        documents = load_pdf_documents(doc_dir=doc_dir, base_dir=BASE_DIR)
        parents, chunking_stats = None, None
        encoding_embeddings = embeddings
        if chunking_config is None:
            text_splitter = RecursiveCharacterTextSplitter(**SPLIT_CONFIG)
            splits = text_splitter.split_documents(documents)
        else:
            from .chunking import build_chunks
            from .embedding_cache import CachedEmbeddings
            from .structure import read_outlines

            encoding_embeddings = CachedEmbeddings(
                embeddings, cache_dir.parent / "embedding_texts.sqlite",
                json.dumps(EMBEDDING_CONFIG, sort_keys=True), BATCH_SIZE
            )
            chunks = build_chunks(
                documents, encoding_embeddings, token_counter, chunking_config,
                read_outlines(documents, BASE_DIR)
            )
            splits, parents, chunking_stats = chunks.children, chunks.parents, chunks.stats
            print("优化切分：" + json.dumps(chunking_stats, ensure_ascii=False))

        if not splits:
            raise ValueError("没有生成可入库的文本片段")
        _report(splits)

        corpus_digest = corpus_hash(splits)
        vectors = get_vectors(encoding_embeddings, splits, corpus_digest, cache_dir, chunking_config)
        if vectors.ndim != 2 or vectors.shape[0] != len(splits):
            raise ValueError("向量数量与文本片段数量不一致，请检查缓存")
        if not np.isfinite(vectors).all():
            raise ValueError("向量中出现 NaN 或无穷值")

        # 保存原始向量；检索器使用归一化后的向量。
        vectors_raw = vectors.copy()
        _normalize(vectors)
        hnsw_index = build_faiss_index(vectors)
        hnsw_retriever = build_hnsw_retriever(splits, hnsw_index, embeddings)
        bm25_retriever = build_bm25_retriever(splits)
        cache_key = make_cache_key(corpus_digest, len(splits), chunking_config)
        if chunking_stats is not None:
            cache_key["chunking_stats"] = chunking_stats

    ensemble_retriever = build_ensemble_retriever(hnsw_retriever, bm25_retriever)
    if from_cache:
        _report(splits)
    else:
        store.save(
            cache_dir=cache_dir,
            fingerprint=fingerprint,
            cache_key=cache_key,
            splits=splits,
            vectors=vectors_raw,
            hnsw_index=hnsw_index,
            bm25_retriever=bm25_retriever,
            parents=parents,
        )
        print(f"已落盘索引到 {cache_dir}（下次启动直接读回，不再解析 PDF）")
        parents = parents or {}

    return IndexBundle(
        splits=splits,
        vectors=vectors,
        hnsw_index=hnsw_index,
        hnsw_retriever=hnsw_retriever,
        bm25_retriever=bm25_retriever,
        ensemble_retriever=ensemble_retriever,
        cache_key=cache_key,
        fingerprint=fingerprint,
        from_cache=from_cache,
        parents=parents,
        token_counter=token_counter,
    )
