"""交叉编码器重排：把召回池按"这段到底回不回答这个问题"重新排序。

两段式检索的标准做法：

    粗排（召回池，两路各 TOP_K 条）→ 精排（cross-encoder 逐条打分）→ 只留 RERANK_TOP_N 条给 LLM

为什么光靠 embedding 不够：bi-encoder 把 query 和文档**分别**编码成向量，两者直到最后
算余弦相似度才见面，所以它只能衡量"语义相近"，判断不了"这段到底回不回答这个问题"。
cross-encoder 把 [query, passage] 拼成一对送进同一个 Transformer 做完整注意力计算，
直接输出一个相关性分数——准得多，代价是必须逐条算、没法预编码，所以只能用在已经
缩小到几十条的候选集上。

本模块只依赖 transformers + torch，不引入 sentence_transformers / FlagEmbedding：
交叉编码器用 AutoModelForSequenceClassification 十几行就够，为它引一个大库不划算。
"""
from __future__ import annotations

import math
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from .telemetry import document_snapshot, document_id

from .config import (
    RERANK_BATCH_SIZE,
    RERANK_DEVICE,
    RERANK_ENABLED,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_TOP_N,
)


# ========== 1. 打分器 ==========
class CrossEncoderScorer:
    """bge-reranker 系列的交叉编码器。

    **权重懒加载**：torch / transformers 的 import 和权重读盘都推迟到 warmup() 里，
    模块导入本身不碰它们。这样 RAG_RERANK=0 时连 torch 都不会被 import，
    启动时间回到原来的水平。

    rag.py 会主动调一次 warmup() 提前加载，让这份耗时出现在启动日志里，
    而不是悄悄算进第一道题的检索时间。
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: str | None = RERANK_DEVICE,
        max_length: int = RERANK_MAX_LENGTH,
        batch_size: int = RERANK_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._requested_device = device
        self._device = None
        self._tokenizer = None
        self._model = None
        # 已经加载过就记住耗时，warmup() 重复调用时返回同一个数
        self.load_seconds = 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    def warmup(self) -> float:
        """把权重加载进显存/内存，返回加载耗时（秒）。已经加载过就直接返回上次的耗时。

        这是幂等的，score() 内部也会先调它，所以忘了预热也只是把耗时挪到第一题上。
        """
        if self.loaded:
            return self.load_seconds

        # 提示放在这里而不是调用方：默认是懒加载，第一次提问时这段是唯一说明
        # "这 16 秒不是卡死"的信息。调用方再打一遍就重复了。
        print(f"正在加载重排模型 {self.model_name}（约 15 秒，只加载一次）…")

        # 计时从**这里**开始，把下面两个 import 也算进去：它们本身就要 7 秒左右
        # （实测 import torch ≈ 4.6s，import transformers ≈ 2.6s）。放在计时之外
        # 会报出一个比实际等待短得多的数字，那种"好看但不对"的日志比没有还糟。
        started = time.perf_counter()

        # 在这里 import 而不是文件顶部：见类文档。已经在 sys.modules 里之后，
        # 再写一次 import 只是一次字典查找，可以忽略。
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        # GPU 上用 fp16：权重从 1.1GB 降到 0.55GB，6GB 的卡留足余量给激活值。
        # CPU 上 fp16 未必有加速，用 fp32 更稳。
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        # 先只认本地缓存。默认行为每次加载都要联网查一遍有没有新版本，实测白花 3.5 秒
        # （tokenizer 3.84s→2.49s，模型 5.43s→3.28s），纯属浪费。
        # 本地没有缓存时（第一次运行）才联网下载，把这段耗时放在明处。
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, dtype=dtype, local_files_only=True
            )
        except Exception:
            # 第一次运行会走这里。权重约 1.1GB，只需下载一次；
            # 网络不通就把 HF_ENDPOINT 设成 https://hf-mirror.com
            print(f"本地没有 {self.model_name} 的缓存，正在从 HuggingFace 下载（约 1.1GB，只需一次）…")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, dtype=dtype)

        self._tokenizer = tokenizer
        self._model = model.to(self._device).eval()
        # 首次前向要额外花约 1.8 秒（CUDA 上下文和 kernel 的第一次调用）。
        # 这里先用一对哑输入把它触发掉，免得算到第一道题的检索时间上。
        self._warm_forward()

        self.load_seconds = time.perf_counter() - started
        print(f"重排模型就绪（{self.load_seconds:.1f} 秒，{self._device}）")
        return self.load_seconds

    def _warm_forward(self) -> None:
        """一次极短的前向，只为把 CUDA / 算子库的一次性初始化成本提前触发掉。"""
        import torch

        inputs = self._tokenizer(
            [["预热", "预热"]], padding=True, truncation=True, max_length=8, return_tensors="pt"
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            self._model(**inputs)

    def score(self, query: str, passages: List[str]) -> List[float]:
        """给每条候选打一个 0~1 的相关性分，顺序和传入的 passages 一一对应。"""
        if not passages:
            return []
        self.warmup()

        import torch

        scores: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            # 交叉编码器的输入是 [query, passage] 这一对，不是两段独立文本
            pairs = [[query, text] for text in batch]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                # only_second 而不是默认的 longest_first：默认行为在 query+正文超长时会把
                # **查询**也截掉一段，那正好截掉了判断相关性最要紧的部分。只截正文。
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            # 先搬到 CPU 再转成 Python 列表，后面的数学用纯 Python 做：
            # 这样 logits_to_scores 不依赖 torch，单测不用为了测一个 sigmoid 付 5 秒 import。
            scores.extend(logits_to_scores(logits.float().cpu().tolist()))
        return scores


def logits_to_scores(rows: List[List[float]]) -> List[float]:
    """把 cross-encoder 的原始 logits 变成 0~1 的分数，越大越相关。

    bge-reranker-base 的 num_labels 是 1，只输出一个 logit，过 sigmoid 即可
    （FlagEmbedding 和 sentence-transformers 的 CrossEncoder 都是这么做的）。
    两个 logit 的模型（有些 reranker 是二分类头）就 softmax 后取正类。
    """
    if not rows:
        return []
    width = len(rows[0])
    if width == 1:
        return [_sigmoid(row[0]) for row in rows]
    if width == 2:
        return [_softmax_positive(row) for row in rows]
    raise ValueError(f"不认识的 logits 形状：每条 {width} 个值")


def _sigmoid(value: float) -> float:
    """分正负两支，避免 exp() 在 |value| 很大时溢出。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softmax_positive(row: List[float]) -> float:
    """softmax 后取第 2 列（正类），减最大值防溢出。"""
    shift = max(row)
    exponentials = [math.exp(value - shift) for value in row]
    return exponentials[1] / sum(exponentials)


# ========== 2. 重排检索器 ==========
def dedupe_documents(documents: List[Document]) -> List[Document]:
    """按 (来源, 页码, 正文) 去重，保持原来的先后顺序。

    EnsembleRetriever 自己也会去重，但它按 page_content 去重：同一段正文出现在
    不同书上时会被合成一条。这里把来源和页码也纳入 key，免得跨书的相同片段被吃掉。
    """
    seen = set()
    result = []
    for doc in documents:
        key = document_id(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


class RerankingRetriever(BaseRetriever):
    """
    把"召回 → 打分 → 截断"封成一个普通 Retriever。
    """

    base_retriever: object
    # 只要实现了 score(query, passages) -> list[float] 就行，不要求是 CrossEncoderScorer
    scorer: object
    top_n: int = RERANK_TOP_N

    @property
    def k(self) -> int:
        """给评测脚本读：evaluate.py 用 getattr(retriever, "k", None) 记进报告。

        对外承诺的是"最终交给 LLM 几条"，不是召回池大小——所以是 top_n 而不是 TOP_K。
        """
        return self.top_n

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        config = {"callbacks": run_manager.get_child()} if run_manager else None
        return self.retrieve_with_trace(query, config=config)[0]

    def retrieve_with_trace(self, query: str, config=None):
        """返回本次调用的候选/重排记录；并发请求之间不共享状态。"""
        started = time.perf_counter()
        documents = dedupe_documents(list(self.base_retriever.invoke(query, config=config)))
        recall_ms = (time.perf_counter() - started) * 1000

        if not documents:
            return [], {"candidates": [], "selected": [], "recall_ms": recall_ms, "rerank_ms": 0.0}

        started = time.perf_counter()
        scores = self.scorer.score(query, [doc.page_content for doc in documents])
        if len(scores) != len(documents) or not all(math.isfinite(s) for s in scores):
            raise ValueError("重排分数数量不匹配或包含非有限值")
        rerank_ms = (time.perf_counter() - started) * 1000
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)

        result = []
        for index in order[: self.top_n]:
            doc = documents[index]
            # 用 model_copy 换一份新的 metadata，而不是就地改 doc.metadata：
            # HNSWRetriever 返回的 Document 和 splits 里的**是同一批对象**，
            # 就地写会把 rerank_score 带进落盘的 splits.jsonl。
            # 而且这个分数依赖当前 query，比 metadata["score"] 更不该被持久化。
            # 同时保留 HNSW 写进去的 metadata["score"]（余弦相似度），两种分数并存
            # 才看得出重排到底改了什么。
            result.append(
                doc.model_copy(
                    update={"metadata": {**doc.metadata, "rerank_score": float(scores[index])}}
                )
            )
        return result, {
            "recall_ms": recall_ms, "rerank_ms": rerank_ms,
            "candidates": [{**document_snapshot(doc, i), "rerank_score": float(scores[i - 1])}
                           for i, doc in enumerate(documents, 1)],
            "selected": [document_snapshot(doc, i) for i, doc in enumerate(result, 1)],
        }


# ========== 3. 工厂：配置只在这里读 ==========
def make_scorer(enabled: bool | None = None) -> CrossEncoderScorer | None:
    """按配置造打分器。enabled=False 时返回 None（= 只用截断，不重排）。"""
    if enabled is None:
        enabled = RERANK_ENABLED
    return CrossEncoderScorer() if enabled else None


def build_reranking_retriever(
    base_retriever,
    scorer,
    top_n: int | None = None,
):
    """包一层重排。参数默认值都从 config 取，配置只在 make_scorer() 里读一次。

    scorer 为 None 时**原样返回 base_retriever**：RAG_RERANK=0 走的就是这条路。
    关掉开关就该和不加重排时完全一样，不留任何残留影响
    （包括"顺手帮你截断到 5 条"——那种半开半关最容易被误读成"关掉后效果反而更差"）。
    """
    if scorer is None:
        return base_retriever
    return RerankingRetriever(
        base_retriever=base_retriever,
        scorer=scorer,
        top_n=RERANK_TOP_N if top_n is None else top_n,
    )
