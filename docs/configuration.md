# 配置说明

运行配置集中在 [`rag_core/config.py`](../rag_core/config.py)，环境变量模板为 [`.env.example`](../.env.example)。模型工厂、索引和 Agent 共同读取模块级常量；本次整理保留 v2 的默认值与所有公开配置名称。

## 配置方式与优先级

首次运行，在项目根目录复制模板：

```powershell
Copy-Item .env.example .env
```

已有 `.env` 时直接编辑原文件，保留现有密钥。加载顺序沿用原实现：通过 `find_dotenv()` 查找 `.env`，以 `override=True` 加载，因此同名配置的优先级为 **`.env` > 系统环境变量 > 代码默认值**。所有启动命令均从项目根目录执行。

布尔开关使用 `0` / `1`；实现按是否等于字符串 `0` 判断，不能用 `false` 代替关闭。设备字段留空时自动选择 CUDA 或 CPU。JSON 报告记录配置名称、模型名与参数，不应写入密钥。

## 问答与裁判

| 配置 | 默认值 / 用途 |
|---|---|
| `ZHIPUAI_API_KEY` | 无默认值；交互问答使用的智谱凭据 |
| `RAG_CHAT_MODEL` | `glm-4-flash`；由 `ChatZhipuAI` 创建 |
| `RAG_LLM_TEMPERATURE` | `0.1` |
| `DEEPSEEK_API_KEY` | 无默认值；选用 DeepSeek 裁判时需要 |
| `DEEPSEEK_API_BASE` | `https://api.deepseek.com`；可设为用户已配置的兼容服务地址 |
| `--judge-model` | 命令行参数；不指定时裁判复用问答模型，历史对照显式使用 `deepseek-flash` |

问答模型与裁判分别配置。当前问答工厂使用智谱，不能只把 `RAG_CHAT_MODEL` 改成其他供应商名称就切换服务；DeepSeek 由评测器的裁判分支调用。裁判模型名需与实际接口支持的名称一致。

## 本地 Embedding

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | 本地中文 BGE |
| `RAG_EMBEDDING_REVISION` | `f03589ceff5aac7111bd60cfc7d497ca17ecac65` | 固定模型快照 |
| `RAG_EMBEDDING_DEVICE` | 自动 | `cpu`、`cuda` 等 |
| `RAG_BATCH_SIZE` | `8` | 文档编码批大小 |

编码设置为 CLS pooling、L2 归一化、float32、最大长度 512 tokens；查询添加检索指令，文档不添加。这些配置在 `EMBEDDING_CONFIG` 中集中定义，并参与缓存指纹。模型权重存放于 `model_cache/huggingface/`。更换模型时需同时核对 revision，不能混用不同模型的文档向量和查询向量。

## 检索与重排

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `RAG_TOP_K` | `20` | HNSW 和 BM25 各自召回的候选数量 |
| `RAG_RERANK` | `1` | 启用交叉编码器重排 |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-base` | 重排模型 |
| `RAG_RERANK_TOP_N` | `5` | 重排后保留的子块数 |
| `RAG_RERANK_MAX_LENGTH` | `512` | 查询与片段合计的模型输入 token 上限 |
| `RAG_RERANK_BATCH_SIZE` | `16` | 重排批大小 |
| `RAG_RERANK_DEVICE` | 自动 | 留空自动选择 |
| `RAG_RERANK_WARMUP` | `0` | 是否在导入 `rag` 时预热；评测也可传 `--warmup-rerank` |

代码常量保持原值：RRF 权重 `ENSEMBLE_WEIGHTS=[0.7, 0.3]`，顺序为向量检索、BM25；HNSW 的 `M=32`、`efConstruction=200`、`efSearch=64`。这些常量未增加新的环境变量入口。调整构图参数时应检查索引失效策略，不能假设旧索引自动使用新参数。

`TOP_K` 控制候选池，`RERANK_TOP_N` 控制重排结果；父块扩展与多轮合并后的最终上下文数量还受后续预算限制。

## 切分与上下文

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `RAG_CHUNK_STRATEGY` | `semantic` | `legacy` / `structural` / `semantic` |
| `RAG_CHUNK_TOKENIZER` | 同重排模型 | 用于切分和上下文长度计量 |
| `RAG_CHUNK_TOKENIZER_REVISION` | 未指定 | 可固定 tokenizer 快照 |
| `RAG_CHILD_MAX_TOKENS` / `RAG_CHILD_MIN_TOKENS` | `320` / `64` | 子块 token 上限与目标下限 |
| `RAG_PARENT_MAX_TOKENS` | `1024` | 父块 token 上限 |
| `RAG_EMBEDDING_MAX_CHARS` | `480` | 子块附加字符上限；字符数不等于 token 数 |
| `RAG_SEMANTIC_PERCENTILE` | `90` | 邻句语义距离切分阈值的分位数 |
| `RAG_SEMANTIC_BUFFER` | `1` | 邻句语义窗口范围 |
| `RAG_PARENT_EXPANSION` | `1` | 子块重排后扩展父块 |
| `RAG_MAX_CONTEXT_TOKENS` | `4096` | 格式化上下文预算 |
| `RAG_MAX_EVIDENCE_DOCS` | `15` | 多轮保留的证据条数上限 |

`legacy` 策略仍使用 `SPLIT_CONFIG` 的 500 字符切分、50 字符重叠。Agent 补搜上限 `MAX_RETRIES=2`，最多在初次检索后补搜两轮。预算使用配置的 tokenizer 计量，不等于问答模型的精确计费 token 数。

## 目录与缓存

| 路径 | 用途 |
|---|---|
| `knowledge_db/book/` | 用户提供的 PDF，可递归读取子目录 |
| `embed_cache_optimized/<策略>_<配置哈希>/` | 当前本地 BGE 的索引与父子块 |
| `embed_cache_books/` | 旧索引目录，保留兼容常量，不与 BGE 混用 |
| `model_cache/` | 本地模型权重 |
| `evaluation/reports/` | 逐题结果、汇总和报告 |

路径均基于仓库根目录，不含用户机器的绝对路径。PDF、模型权重、索引、密钥和完整运行报告默认不提交。索引格式版本 `SCHEMA_VERSION=2` 保持不变；配置分组和注释整理不会要求重新入库。若修改切分算法、模型或语料，应按缓存指纹重新准备相应索引。
