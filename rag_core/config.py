"""运行配置；环境变量示例见 .env.example，完整说明见 docs/configuration.md。

保持模块级常量与普通字典接口，供问答、索引和评测模块共同读取。
.env 覆盖同名系统环境变量；默认值保持已验证的 v2 配置。
"""
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# 路径与环境
BASE_DIR = Path(__file__).resolve().parent.parent
_ = load_dotenv(override=True, dotenv_path=find_dotenv())
DOC_DIR = BASE_DIR / "knowledge_db" / "book"
CACHE_DIR = BASE_DIR / "embed_cache_books"  # 保留旧缓存路径接口，不与本地 BGE 混用。
OPTIMIZED_CACHE_DIR = BASE_DIR / "embed_cache_optimized"
EMBEDDING_CACHE_DIR = BASE_DIR / "model_cache" / "huggingface"

# 问答模型（ChatZhipuAI）；Judge 由 evaluation/evaluate.py 的命令行参数选择。
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "glm-4-flash")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))

# 本地 BGE：模型版本与编码设置参与索引指纹。
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
EMBEDDING_REVISION = os.getenv("RAG_EMBEDDING_REVISION", "f03589ceff5aac7111bd60cfc7d497ca17ecac65")
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE") or None
BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "8"))
EMBEDDING_CONFIG = {
    "backend": "local_bge",
    "version": 1,
    "model": EMBEDDING_MODEL,
    "revision": EMBEDDING_REVISION,
    "pooling": "cls",
    "normalize": True,
    "dtype": "float32",
    "max_length": 512,
    "query_instruction": "为这个句子生成表示以用于检索相关文章：",
}

# 召回与融合：TOP_K 是每路候选数，权重顺序为向量检索、BM25。
TOP_K = int(os.getenv("RAG_TOP_K", "20"))
ENSEMBLE_WEIGHTS = [0.7, 0.3]
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# 重排：关闭时直接返回融合检索结果；设备留空表示自动选择。
RERANK_ENABLED = os.getenv("RAG_RERANK", "1") != "0"
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))
RERANK_MAX_LENGTH = int(os.getenv("RAG_RERANK_MAX_LENGTH", "512"))
RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", "16"))
RERANK_DEVICE = os.getenv("RAG_RERANK_DEVICE") or None
RERANK_WARMUP = os.getenv("RAG_RERANK_WARMUP", "0") != "0"

# 切分：legacy 为逐页字符切分；structural / semantic 使用结构父块。
SPLIT_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", "。", "!", "?", " ", ""],
}
CHUNK_STRATEGY = os.getenv("RAG_CHUNK_STRATEGY", "semantic").lower()
CHUNKING_CONFIG = {
    "version": 1,
    "strategy": CHUNK_STRATEGY,
    "tokenizer": os.getenv("RAG_CHUNK_TOKENIZER", RERANK_MODEL),
    "tokenizer_revision": os.getenv("RAG_CHUNK_TOKENIZER_REVISION") or None,
    "child_max_tokens": int(os.getenv("RAG_CHILD_MAX_TOKENS", "320")),
    "child_min_tokens": int(os.getenv("RAG_CHILD_MIN_TOKENS", "64")),
    "parent_max_tokens": int(os.getenv("RAG_PARENT_MAX_TOKENS", "1024")),
    "embedding_max_chars": int(os.getenv("RAG_EMBEDDING_MAX_CHARS", "480")),
    "semantic_percentile": float(os.getenv("RAG_SEMANTIC_PERCENTILE", "90")),
    "semantic_buffer": int(os.getenv("RAG_SEMANTIC_BUFFER", "1")),
}

# Agent 与上下文：父块扩展发生在子块重排之后。
PARENT_EXPANSION = os.getenv("RAG_PARENT_EXPANSION", "1") != "0"
MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "4096"))
MAX_EVIDENCE_DOCS = int(os.getenv("RAG_MAX_EVIDENCE_DOCS", "15"))
MAX_RETRIES = 2

# 索引格式版本。修改未被自动指纹覆盖的构库逻辑时需评估是否递增。
# 仅整理配置分组不改变缓存格式，因此保留原值。
SCHEMA_VERSION = 2
