"""Agentic RAG 核心模块。

config / models：配置与模型工厂。
structure / chunking / provenance / token_budget：结构与语义切分、来源定位、预算。
local_embeddings / embedding_cache：本地 BGE 编码与文本向量缓存。
store / indexing：持久化索引、HNSW 与 BM25 / RRF 召回。
rerank / parent_retriever：交叉编码器重排与父块扩展。
prompts / structured / graph / telemetry：提示词、解析、流程和节点追踪。

导入核心模块不会自动构建索引；调用 build_index() 或导入 rag 时才准备索引。
"""
