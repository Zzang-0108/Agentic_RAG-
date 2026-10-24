# Agent 流程回退记录

2026-10-24：根据开发集退化结果，将运行代码从 v3 回退到 `run_20261022T051059_048567Z` 对应的 v2。当前设计见 [三路路由、整组证据检查与补搜](agent_workflow.md)。

## 回退依据与范围

历史报告中，v2 Agent 在同一份 120 题开发集上由 `deepseek-flash` 裁判判对 99 题（82.5%）；v3 判对 37 题（30.8%），内部流程失败 69 题。详细对比及归因边界见 [失败案例复盘](agentic_rag_v3_failure_case.md)。82.5% 是历史结果，本次未重新调用在线模型评测，不能将其表述为回退后新测成绩。

恢复源为 `.work_backups/workflow_v3_before_20261022_154002/`。已逐文件核对备份清单和历史评测中的 SHA-256，恢复以下 9 个文件：

- `rag.py`
- `rag_core/graph.py`、`rag_core/prompts.py`、`rag_core/structured.py`、`rag_core/config.py`
- `evaluation/evaluate.py`、`evaluation/test_evaluate.py`
- `tests/test_graph.py`、`tests/test_decisions.py`

v3 专用的 `rag_core/evidence.py` 与 `tests/test_evidence.py` 已归档并移出活动目录。回退完成时，20 个运行源文件及评测脚本哈希均与 v2 报告一致。

同日随后按用户要求进行 GitHub 仓库整理：重新分组 `config.py`、清理 `rag.py` 的废弃注释并格式化入口、精简包说明，因而这三个文件的文本哈希随后变化。已核对入口可执行语法树不变、配置赋值表达式与 32 个默认常量值不变，其余 16 个核心模块逐字节不变。配置使用说明见 [configuration.md](configuration.md)；该整理不是新的算法版本。

后续代码简化：合并文档去重逻辑，展开格式化与结构化输出解析中的嵌套表达式，直接配对文档和重排分数，并统一缓存加载与新建索引的返回路径。此次调整涉及 `graph.py`、`structured.py`、`telemetry.py`、`rerank.py` 和 `indexing.py`，不再满足上述整理阶段的逐字节相同条件；配置、提示词与切分实现保持不变。120 项核心测试、48 项评测测试通过，另有 20 个场景的模型输入、查询序列、最终状态和去除耗时后的 trace 与调整前完全一致。没有运行新的在线质量评测。

| 环节 | 当前状态 |
|---|---|
| 本地 BGE、FAISS HNSW、BM25 + RRF、rerank | 保留 |
| 结构父块、语义子块、父块扩展、上下文预算 | 保留 |
| 检索 / 直接回答 / 澄清三路路由 | 恢复 v2 实现 |
| 整组证据充分性检查与定向补搜 | 恢复 v2；保留有限重试、重复查询与预算停止 |
| 生成回答 | 恢复自然语言生成，提示词要求引用资料并说明缺失内容 |
| 固定待答要点、逐点原文摘录强校验、结构化答案强校验、无覆盖进展停止 | 随 v3 撤下 |
| 普通 RAG 对照入口及 `--mode compare` / `rag` | 随 v3 撤下 |
| 320 题评测集、已有报告与失败案例 | 保留 |

v2 仍对路由和证据检查的 JSON 输出进行 Pydantic 校验，检查状态一致性与证据编号范围。回退没有取消所有格式校验，也没有回到逐篇调用 LLM 筛选的早期实现。

## 备份与数据保护

回退前的代码、测试和本次涉及的文档存放在：

`.work_backups/workflow_v3_rollback_20261024T064422Z/`

同目录的 `manifest.json` 保存了回退前文件哈希、恢复文件哈希、索引哈希、数据集哈希和两轮历史报告哈希。备份未包含 API 密钥。失败实验仍可复查；其设计文档已标明为历史版本。

现有 `embed_cache_optimized/semantic_71669aa50ac9/` 的 6 个索引文件哈希未变，数据集及两轮历史报告哈希未变。无需重新入库。此次没有修改 `.env`、模型依赖、用户笔记本或其他不相关改动。

## 验证记录

- 核心离线测试：118 项通过。
- 评测脚本离线测试：48 项通过。
- 恢复代码与历史报告哈希核对：在回退完成时通过；后续文档与格式整理的变化见上文。
- 真实本地检索与假 LLM 接线检查：通过；BGE 与 reranker 均使用 CUDA，完成 route → retrieve → assess → generate。
- 直接加载现有 768 维 HNSW 索引，4676 个子块、2006 个父块；没有重新解析 PDF 或编码文档，调用前后索引哈希一致。
- 本次检查未调用付费问答或裁判接口。详细结果见本地 `.work_backups/workflow_v2_rollback_integration.json`。

离线测试验证代码行为与故障处理，不代表真实模型的回答正确率。测试日志中的临时索引构建属于测试用例，不是重建用户知识库。

## 当前运行命令

交互问答：

```powershell
python rag.py
```

再次评测 120 题开发集上的 Agent：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --mode agent --judge --judge-model deepseek-flash --warmup-rerank
```

如需连同 BM25、dense、hybrid、rerank 一起评测，将 `--mode agent` 改成 `--mode both`，共 600 条记录。当前仅支持 `retrieval`、`agent`、`both`，不要使用已撤回的 `compare` / `rag`。不显式指定数据集时仍默认使用 6 题 smoke 集。

以上在线问答和裁判命令会调用已配置的模型接口，本次回退没有执行它们。原报告目录保留，新评测会生成新的时间戳目录。终端或 Notebook 若已经导入过 `rag`，请重启该 Python 进程后再运行，以加载恢复后的代码。
