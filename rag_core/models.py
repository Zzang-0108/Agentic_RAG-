"""模型工厂：换模型只改这里。"""
from langchain_community.chat_models import ChatZhipuAI
from .config import (
    CHAT_MODEL, LLM_TEMPERATURE, EMBEDDING_CONFIG, EMBEDDING_DEVICE,
    EMBEDDING_CACHE_DIR, BATCH_SIZE,
)
from .local_embeddings import LocalBGEEmbeddings


def make_llm() -> ChatZhipuAI:
    """对话模型。"""
    return ChatZhipuAI(model=CHAT_MODEL, temperature=LLM_TEMPERATURE)


def make_embeddings() -> LocalBGEEmbeddings:
    """本地 BGE：语义窗口、入库文档、检索问题共用同一个模型。"""
    embeddings = LocalBGEEmbeddings(
        model_name=EMBEDDING_CONFIG["model"], revision=EMBEDDING_CONFIG["revision"],
        device=EMBEDDING_DEVICE, cache_dir=EMBEDDING_CACHE_DIR, batch_size=BATCH_SIZE,
        max_length=EMBEDDING_CONFIG["max_length"],
        query_instruction=EMBEDDING_CONFIG["query_instruction"],
    )
    # 命中落盘索引时也预热，避免首道评测题包含模型加载耗时。
    embeddings.warmup()
    return embeddings
