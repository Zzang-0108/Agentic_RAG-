# 结构切分、Semantic Splitter 与父子块检索

2026-10-21：三个优化已接入 `rag.py`，默认策略为 `semantic`。这次只做代码与离线功能检查，没有运行开发集、测试集或在线模型裁判，也没有调用线上 embedding 构建正式的新索引。因此目前没有新策略的准确率、召回率或延迟结论。

## 实际流程

```mermaid
flowchart LR
    A[逐页 PDF 文本与书签] --> B[清理页眉页脚与目录\n恢复章节和跨页正文]
    B --> C[有长度上限的结构父块]
    C --> D[邻句窗口 embedding\n语义边界与长度约束]
    D --> E[子块 HNSW 与 BM25]
    E --> F[RRF 融合与子块 rerank]
    F --> G[按 parent_id 扩展父块\n去重与上下文预算]
    G --> H[证据充分性检查与回答]
```

1. **结构切分**：优先使用能够在正文中定位的 PDF 书签和编号标题，识别跨行标题，连接同一章节的跨页正文。过滤明显目录、重复页眉页脚和装饰标记。不同书籍或缺失页之间不拼接。
2. **有长度约束的语义切分**：在父块内部划分中文句段，为有界邻句窗口计算 embedding，用相邻窗口的余弦距离分位数寻找话题边界，再按长度约束组合为子块。过短尾块在不超限时合并；超过硬上限的单句、代码或公式会强制切开并标记，正文不直接丢弃。
3. **父子块检索**：向量检索、BM25 和 rerank 都处理子块，只有重排完成后才补回父块。命中同一父块的多个子块合并为一条上下文；预算不足时尝试放入实际命中的子块，不只截取父块开头。

这里实现了项目内的 semantic splitter，没有额外引入 `langchain-experimental`。句段划分、长度限制和字符位置映射在同一条流程中完成。

## 构建与启动

在项目根目录运行，使用已有的 `torch_gpu` 环境。当前环境检查发现 `jsonpointer` 无法正常导入；已在 `requirements.txt` 中补充依赖，先安装这一项：

```powershell
conda activate torch_gpu
python -m pip install jsonpointer==2.1
python -m rag_core.prepare_index --strategy semantic
python rag.py
```

`prepare_index` 只建库，不执行问答评测。2026-10-22 起默认使用本地 BGE-base 中文版，分别计算语义窗口与最终子块的向量，无 Embedding API 费用；首次需要下载权重，详见 [本地 BGE 配置](local_bge.md)。父块不单独计算检索向量。批次成功后立即写入文本向量缓存；中途失败时，重新运行同一命令可复用已完成的文本。

`rag.py` 默认同样使用 `semantic`。如果 `.env` 中已经设置了其他 `RAG_CHUNK_STRATEGY`，请与构建策略保持一致。命令行 `--strategy` 只影响本次建库，不修改 `.env`。项目当前的 `load_dotenv(override=True)` 会让 `.env` 覆盖同名 shell 环境变量。

切分长度使用本地缓存的 `BAAI/bge-reranker-base` tokenizer。当前机器已具备该缓存；换机器时需提前准备 tokenizer，或把 `RAG_CHUNK_TOKENIZER` 设置为本地 tokenizer 目录。加载器不会在离线预览中尝试联网下载。

只检查结构、查看实际父子块时，可运行：

```powershell
python -m rag_core.prepare_index --preview-structure
```

这条命令读取 PDF 与本地 tokenizer，写出 `parents.jsonl`、`children.jsonl` 和 `summary.json`，核对每段来源文本；不创建模型接口、不计算语义向量、不构建检索索引。预览会强制使用 `structural`，即使当前默认策略是 `semantic`。`--output` 可指定一个尚不存在的预览目录。

## 配置

主要参数在 `rag_core/config.py`，可写入 `.env`：

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `RAG_CHUNK_STRATEGY` | `semantic` | `semantic` 为结构＋语义；`structural` 只按结构和长度划分；`legacy` 为原逐页切分 |
| `RAG_CHILD_MAX_TOKENS` | `320` | 检索子块正文 token 上限 |
| `RAG_CHILD_MIN_TOKENS` | `64` | 语义边界处的最小长度目标；章节末尾和强制切分时不保证达到 |
| `RAG_EMBEDDING_MAX_CHARS` | `480` | 子块与语义窗口的附加字符上限 |
| `RAG_PARENT_MAX_TOKENS` | `1024` | 单个父块正文 token 上限 |
| `RAG_SEMANTIC_PERCENTILE` | `90` | 父块内相邻窗口距离的分位数阈值，距离严格超过阈值才作为候选边界 |
| `RAG_SEMANTIC_BUFFER` | `1` | 每侧最多纳入的邻句数，同时受长度限制 |
| `RAG_PARENT_EXPANSION` | `1` | 重排后是否扩展父块；设为 `0` 则保留子块 |
| `RAG_MAX_CONTEXT_TOKENS` | `4096` | 每轮合并后、包含资料标题和分隔符的上下文预算 |
| `RAG_CHUNK_TOKENIZER` | `BAAI/bge-reranker-base` | 切分与上下文预算的计数 tokenizer，也支持本地目录 |

这些默认值用于先建立可运行的优化版本，尚未通过问答效果评测选优。320-token 子块为 reranker 的查询及特殊 token 留出空间，但极长查询仍可能导致截断。计数使用 BGE tokenizer，不等于 `embedding-2` 或生成模型的精确 token 数；4096 预算也不包含系统提示、用户问题和输出预留。

切回原切分方式，在 `.env` 设置 `RAG_CHUNK_STRATEGY=legacy` 后启动即可。该策略使用逐页 500 字符、50 字符 overlap 切分，但仍使用当前 BGE 模型重新建索引，不复用旧智谱向量。

## 来源与父子关系

`children.jsonl` / 索引的 `splits.jsonl` 每行是一条子块，`parents.jsonl` 每行是一条父块。两者均包含 `page_content` 与 `metadata`。

| 字段 | 含义 |
|---|---|
| `source`、`book_name` | 原 PDF 路径与书名标识 |
| `page_number` | 实际包含的第一个 PDF 页号，兼容旧调用方；从 1 开始 |
| `page_numbers` | 该块实际包含的所有 PDF 页号，不把首尾之间所有页面都算进去 |
| `source_spans` | 块内字符区间与原始页文本字符区间的一一映射 |
| `section_title`、`section_path` | 从标题或书签推断的章节路径；未识别时可为空 |
| `chunk_id` | 由来源、正文和来源位置生成的稳定块标识 |
| `chunk_type` | `child` 或 `parent` |
| `parent_id` | 对应父块标识 |
| `parent_start`、`parent_end` | 子块在父块正文内的位置，左闭右开 |
| `token_count` | 配置 tokenizer 对正文的计数 |
| `forced_split` | 该块或所属父块的原子段曾因硬长度上限被切开 |
| `matched_children` | 查询期间命中的子块及其重排分数、原检索分数和父块内位置 |

每条 `source_spans` 形如 `{start, end, page_number, page_start, page_end}`。它表示 `page_content[start:end]` 与原 PDF 提取文本的 `[page_start:page_end]` 完全一致。整理时加入的空格或换行不冒充 PDF 原文字符。页号是 PDF 文件页序，不是书内印刷页码。

子块的 `rerank_score` 不冒充父块得分；父块通过 `matched_children` 保留这些分数。查询 trace 另存 `parent_expansion` 和 `context_budget`，可查看实际保留的父块、预算回退子块和被省略的块。

## 缓存

- `embed_cache_books/`：原有 legacy 索引，保留不覆盖。
- `embed_cache_optimized/<策略>_<配置哈希>/`：当前策略的父子块、向量、HNSW、BM25 与元数据，哈希包含模型编码配置。
- `embed_cache_optimized/embedding_texts.sqlite`：按 embedding 模型、权重版本、编码方式与完整文本缓存向量，支持跨切分参数复用相同文本和失败后续跑。

索引指纹包含 PDF、加载器、切分源码、切分配置和 tokenizer 签名。恢复时检查父子块文件哈希、块数量和父子定位关系，避免误用旧映射。正常加载完整索引不会再次调用语义切分的 embedding。

## 本次离线检查与边界

已通过项目测试 117 项，以及评测工具的离线兼容性测试 48 项。测试使用合成文档、假 embedding 和假 LLM，覆盖语义边界、长度、跨页来源、父块去重、预算回退、缓存中断恢复与旧流程兼容；没有运行真实题目的效果评测。

本地四本 PDF 共读入 1,368 个非空页面。最新结构预览保存在 `embed_cache_optimized/preview_structure_20261021_v3/`：2,006 个父块、4,192 个子块，其中 996 个子块包含跨页正文。所有父子块的来源字符映射均与原始页面文本逐段核对通过。该预览未计算语义 embedding，不能把这些数量当作正式 semantic 索引的数量。

目录、页眉页脚和标题识别仍是启发式规则。PDF 原始抽取仍可能丢失表格结构或公式版式；没有明确代码/公式标记时不能保证识别出完整原子块。长度硬约束也可能留下短尾块，因此没有按“短于某字数”直接删除正文。来源映射校验只能证明保留文本的出处准确，不能证明解析已完整恢复所有版面内容，也不能证明检索效果提升。
