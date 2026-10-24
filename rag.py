"""问答入口与评测接口。

导入时准备本地 BGE 和检索索引；缓存命中后复用已落盘的父子块与向量。
评测使用 llm、app、四个检索器、CHAT_MODEL、EMBEDDING_MODEL、
SPLIT_CONFIG 和 CACHE_KEY；保留原模块级名称以兼容现有调用。
"""
from rag_core.config import (
    BASE_DIR,
    BATCH_SIZE,
    CACHE_DIR,
    CHAT_MODEL,
    DOC_DIR,
    EMBEDDING_MODEL,
    RERANK_WARMUP,
    SPLIT_CONFIG,
    TOP_K,
    CHUNKING_CONFIG,
    PARENT_EXPANSION,
    MAX_CONTEXT_TOKENS,
)
from rag_core.graph import build_app
from rag_core.indexing import build_index
from rag_core.models import make_embeddings, make_llm
from rag_core.rerank import build_reranking_retriever, make_scorer
from rag_core.parent_retriever import ParentContextRetriever

llm = make_llm()
embeddings = make_embeddings()
_index = build_index(embeddings, chunking_config=CHUNKING_CONFIG)

splits = _index.splits
vectors = _index.vectors
hnsw_index = _index.hnsw_index
hnsw_retriever = _index.hnsw_retriever
bm25_retriever = _index.bm25_retriever
ensemble_retriever = _index.ensemble_retriever

# 保留未重排的 ensemble_retriever，供 hybrid 与 rerank 对照。
rerank_scorer = make_scorer()
rerank_retriever = build_reranking_retriever(ensemble_retriever, rerank_scorer)

if rerank_scorer is not None:
    if RERANK_WARMUP:
        rerank_scorer.warmup()
    else:
        print(f"重排已启用（{rerank_scorer.model_name}）：模型将在第一次检索时加载")

CACHE_KEY = _index.cache_key

# 纯检索评测返回子块；Agent 在重排后扩展父块并应用上下文预算。
context_retriever = (
    ParentContextRetriever(
        base_retriever=rerank_retriever,
        parents=_index.parents,
        token_counter=_index.token_counter,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    if PARENT_EXPANSION and _index.parents
    else rerank_retriever
)
app = build_app(
    llm,
    context_retriever,
    token_counter=_index.token_counter,
    max_context_tokens=MAX_CONTEXT_TOKENS,
)


if __name__ == "__main__":
    while True:
        query = input("\n提问（q退出）：").strip()
        if query.lower() == "q":
            break
        result = app.invoke({"question": query})
        print(f"\n回答:{result['generation']}\n")
        print(
            f"(重写次数:{result.get('retries', 0)},"
            f"最终文档数:{len(result.get('documents', []))})"
        )
