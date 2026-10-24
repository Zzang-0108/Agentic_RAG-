# 本地 BGE-base 中文 Embedding

2026-10-22：主程序 `rag.py` 和 `evaluation/evaluate.py` 使用本地 `BAAI/bge-base-zh-v1.5`，语义切分窗口、最终子块及查询均由该模型编码。不会调用智谱 Embedding API；对话模型和 Judge 仍使用原来的配置，费用另计。教程 notebooks 的独立示例未同步切换。

实现遵循 [BAAI 官方模型卡](https://huggingface.co/BAAI/bge-base-zh-v1.5)：取最后一层的 CLS 向量并作 L2 归一化，输出 768 维。查询添加 `为这个句子生成表示以用于检索相关文章：`，文档与语义窗口不添加。最大输入 512 tokens（包含特殊 token 与查询指令），超长查询截断；当前子块另有 480 字符和 320 个 reranker tokens 的限制。

权重固定为 `f03589ceff5aac7111bd60cfc7d497ca17ecac65`，使用 float32 推理。默认自动选择 CUDA，没有 CUDA 时使用 CPU。复用已安装的 torch、transformers，不额外依赖 sentence-transformers 或 FlagEmbedding。

## 构建与运行

在项目根目录和 `torch_gpu` 环境中运行：

```powershell
python -m rag_core.prepare_index --strategy semantic
```

这条命令只建库，不调用对话模型或 Judge。权重首次下载到项目 `model_cache/huggingface/`，之后优先使用本地快照；完整索引命中后也会跳过 PDF 解析和语义切分。加载模型、每处理 100 个父块、编码子块时都会输出进度。

```powershell
python rag.py
```

准备正式运行开发集时使用原命令：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --mode both --judge --judge-model deepseek-flash --warmup-rerank
```

建库和查询端必须使用相同配置。两轮 Agent 改造对比应固定本次 BGE 模型、切分和索引，不能直接把旧智谱评测当作仅修改 Agent 的对照。

## 配置与缓存

默认值在 `rag_core/config.py`。已有 `.env` 若设置了 `RAG_EMBEDDING_MODEL=embedding-2`，需移除该行或改为以下配置：

```dotenv
RAG_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
RAG_EMBEDDING_REVISION=f03589ceff5aac7111bd60cfc7d497ca17ecac65
RAG_BATCH_SIZE=8
```

`RAG_EMBEDDING_DEVICE=cpu` 可强制 CPU；不设置时自动选择设备。批大小只影响资源占用，不改变模型标识。更换其他 BGE 仓库时应同时指定其对应的 revision；本地权重目录发生变化时也须更新 revision，避免复用旧向量。

新索引在 `embed_cache_optimized/<策略>_<配置哈希>/`，哈希与报告 `CACHE_KEY.embedding_config` 记录模型版本、CLS pooling、归一化、精度、长度和查询指令。SQLite 文本缓存使用相同编码配置隔离，旧智谱条目继续保留但不会被 BGE 读取。旧 `embed_cache_books/` 保留；即使选择 `legacy` 切分，BGE 仍在新目录重建自己的索引。

本地模型验证和建库成功只代表链路可用，检索效果需要另行运行开发集评测确认。

## 本机完成记录（2026-10-22）

- 已下载上述固定版本，使用 RTX 3060 Laptop GPU 完成本地编码。
- 四本 PDF 已完成 semantic 建库：2,006 个父块、4,676 个子块，768 维 `IndexHNSWFlat`。
- 索引目录：`embed_cache_optimized/semantic_71669aa50ac9/`，元数据 `complete=true`。
- 新进程重载命中完整索引；检查期间禁止 PDF 解析与文档编码，仍正常完成加载。全部向量有限且单位范数；真实查询通过 HNSW、BM25，并与现有 BGE reranker 在 CUDA 上联合运行。
- 原有 117 项项目测试、48 项评测工具测试通过；新增的模型版本变更失效检查在对应 7 项缓存测试中通过。
- 验证环境：torch `2.13.0+cu126`、transformers `4.57.6`。未运行开发集/测试集效果评测或在线 Judge。

本机检查明细保存在 `.work_backups/bge_validation.json`，建库日志在 `.work_backups/bge_build_index.log`；这些本地产物及模型权重已被 Git 忽略。
