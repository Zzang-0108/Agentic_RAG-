# 现有 Agentic RAG 的自动评测

这个目录提供独立脚本，不修改 `rag.py` 和 `file_loader.py`。

当前使用一套[评测集](datasets/README.md)：开发集 120 题、测试集 200 题，统一存放在 `datasets/`，不再按版本分目录。使用时显式传入 `--dataset evaluation/datasets/dev.jsonl`（或 `test.jsonl`）；不传参数仍使用 6 题 smoke 集。离线格式检查可加 `--validate-only`；完整证据核验记录见 [validation_report.json](datasets/validation_report.json)。指标现状与待补项目见 [评测审查说明](../docs/evaluation_audit_20261022.md)。

被测对象是项目根目录的 `rag.py`（门面，公开名字 `llm` / `app` / 四个检索器 / `CACHE_KEY` 都没变）和它依赖的 `rag_core/` 包：`config`（配置）· `models`（模型工厂）· `store`（索引落盘）· `indexing`（建索引与 HNSWRetriever）· `rerank`（交叉编码器重排）· `prompts` · `graph`（LangGraph 图）。评测脚本读到的状态字段 `need_retrieve` / `retries` / `documents` / `generation` 也是原样保留的；`documents` 是**证据检查后实际用于生成**的列表，`agent` 的页指标口径仍是 `final_context`。

当前已于 2026-10-24 回退到三路路由、整组证据检查与补搜的 v2。流程与日志字段见 [Agent 问答流程 v2](../docs/agent_workflow.md)，代码恢复与离线验证见 [回退记录](../docs/workflow_rollback_20261024.md)。

当前 Agent 评测命令（120 道 dev 题，共 120 条记录；会调用已配置的问答和裁判接口）：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --mode agent --judge --judge-model deepseek-flash --warmup-rerank
```

当前 `--mode` 支持 `retrieval`、`agent`、`both`。`both` 跑四路检索加 Agent，在这份开发集上共 600 条记录。都需要显式 `--dataset`，否则仍是 6 题 smoke 集。v3 的 `compare`、`rag` 与 `rag_baseline_app` 已随版本回退撤下，历史结果仍保存在原报告目录。

2026-10-21 起默认使用[结构父块与语义子块](../docs/chunk_optimization.md)。检索基线返回子块，Agent 上下文可能是扩展父块；跨页片段按 `source_spans` 中实际包含的 PDF 页号计分。父块覆盖更多页面，不能把页命中上升直接解释为具体证据检索变好。本次仅适配序列化和页码处理，未运行开发集或测试集评测。

- `evaluate.py`：批量执行检索或 Agent 问答，计算指标并输出报告。
- `smoke.jsonl`：6 道流程验证题，其中 4 道有书籍依据，另有 1 道信息缺失题和 1 道问候题。
- `test_evaluate.py`：不访问模型接口的指标与集成测试。

示例题只用于试跑；题量和覆盖不足以代表四本书的整体表现，不能直接作为简历效果数据。

## 1. 先校验数据，不调用模型

在项目根目录、原来能运行 `rag.py` 的 Python 环境中执行。Python 需 3.10 或更新版本。

```powershell
python evaluation/evaluate.py --validate-only
```

该命令仅使用标准库，检查 JSONL 格式、重复 ID、标签、页码格式和证据文件存在性；不导入 `rag.py`，不解析 PDF，不核验页码上限或答案内容。

## 2. 比较四种检索方式

```powershell
python evaluation/evaluate.py --mode retrieval --retrievers bm25 dense hybrid rerank --k 5
```

复用当前程序中的 `bm25_retriever`、`hnsw_retriever`、`ensemble_retriever`、`rerank_retriever`。采用相同问题和现有召回配置，不在评测过程中调整其权重或 k。

`hybrid` 是**没有重排**的 RRF 融合结果，`rerank` 是它的候选逐条过交叉编码器（`bge-reranker-base`）后留下的 `RERANK_TOP_N` 条。两者对照才是重排的净效果，别把 `rerank` 和 `dense` 比。

**不写 `--retrievers` 时默认四路全跑**（默认值是从 `RETRIEVERS` 这个字典派生的，加了第四种方法就自动变成四种）。只想要基线就显式写 `--retrievers bm25 dense hybrid`，再配上 `RAG_RERANK=0`——那会连模型都不加载。

`--k` 是评分截断位置，不是底层召回数量。`bm25` / `dense` 各自按自己的 k 召回，传 `--k 10` 不会自动召回 10 条；脚本会提示配置差异。混合检索可能返回更多候选，但评分只看前 K 个片段。

**首次实时运行会导入 `rag.py`，由它自己的初始化逻辑准备索引。** `legacy` 策略使用 `embed_cache_books/`；默认 `semantic` 策略使用 `embed_cache_optimized/semantic_<配置哈希>/`。完整索引命中后跳过 PDF 解析、切分和建图；优化索引还校验父子块文件及其映射。

当前 embedding 使用本地 `BAAI/bge-base-zh-v1.5`，句子窗口、文档和查询均在本地编码，不收取 Embedding API 费用。完整索引命中时不解析 PDF、不重新编码文档；缓存未命中才建库。旧云端模型的向量不能与 BGE 查询向量混用，索引按配置隔离，详见 [本地 BGE](../docs/local_bge.md)。即使只选择 BM25，模块级初始化也会准备向量索引。脚本本身不修改源代码。

### 重排模型的加载时间算在哪

`bge-reranker-base` 实测要 **16.4 秒**才加载好（`import torch` 4.6s + `import transformers` 2.6s + 读 1.1GB 权重和一次哑前向 9.2s）。默认它是**懒加载**的：`import rag` 不受影响，这 16 秒推迟到第一次真正检索时才付。

默认懒加载会把加载成本算进**第一个实际检索样本的延迟**，小样本的 P95 尤其容易受影响。当前脚本支持 `--warmup-rerank`，在样本计时前加载模型并完成首次前向：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --mode retrieval --retrievers bm25 dense hybrid rerank --warmup-rerank
```

加载和预热耗时会计入 `summary.json` 的 `initialization_ms`，不掺进 `latency_ms`。耗时取决于机器和运行状态，上述 16.4 秒只是之前一次测量。命令行预热开关也避免了 `.env` 覆盖同名环境变量带来的歧义。

三个开关（都可以写在 `.env` 里）：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `RAG_RERANK` | `1` | 设 `0` 完全关掉重排，`rerank_retriever` 就是 `ensemble_retriever` 本身，行为和不加重排时一致 |
| `RAG_RERANK_WARMUP` | `0` | 设 `1` 在 `import rag` 时就把模型加载好（评测用） |
| `RAG_RERANK_TOP_N` | `5` | 重排后交给 LLM 的条数 |

## 3. 跑完整 Agent

先用少量题检查链路：

```powershell
python evaluation/evaluate.py --mode agent --limit 2
```

这一步保存实际回答、最终上下文、路由结果、重写次数、耗时和引用编号检查结果。不传 `--judge` 就不会生成回答正确率或忠实性分数。

定向回归可使用 `--ids`，题目仍从原始数据集读取，不另写答案：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --mode agent --ids cd-01 dd-03 p03-3 --judge --judge-model deepseek-flash --warmup-rerank
```

需要自动判断答案质量时，显式启用模型裁判：

```powershell
python evaluation/evaluate.py --mode agent --judge
```

默认复用 `rag.llm`。可以用 `--judge-model` 指定裁判（同时传 `--judge`）。`deepseek-flash` 走已配置的 DeepSeek 接口；其他模型按脚本中的供应商分支处理。当前前后对照应继续显式指定 `deepseek-flash`，保持与 `run_20261022T051059_048567Z` 相同的裁判配置。

以下为早期 GLM 裁判的历史诊断，不是当前换用 GLM 的建议。在当时 dev 集回放同一批记录、只换裁判模型实测：

| 裁判 | 正确拒答的行里 `abstained` 判对 | 解析失败 |
|---|---:|---:|
| glm-4-flash | 9/13 | 0 |
| glm-4-plus | 13/13 | 0 |

当时 glm-4-flash 的部分结果在 `reason` 中认可拒答，却将 `abstained` 填为 false，会使拒答率偏低。历史表只支持这批样本上的比较；更换裁判后仍需抽查，不能将不同裁判得分直接当作流程改进。

使用不同的裁判模型可以减少直接自评带来的部分偏差，但不能消除同系列模型的共同偏差，也不能替代人工核验。

有关模型差异的一处实现细节：glm-4-plus / glm-4-air / glm-4-airx 会把可空字段序列化成字符串（`null` 写成 `"null"`、`true` 写成 `"true"`），`judgment_schema()` 里对此做了受限兼容；`answer_correct` / `abstained` 仍然只接受真布尔值，不接受字符串布尔。

使用同一模型生成和评分可能有自评偏差，应人工抽查，并固定裁判模型、提示词和版本后比较实验。

裁判每题额外调用模型，耗时和 Token 单独保存。结构化输出为空、解析失败等会记录为 `judge_error`，不会把它伪装成 0 分，也不会阻止后续题目运行。

同时评测四种检索与完整 Agent：

```powershell
python evaluation/evaluate.py --mode both --judge
```

若评测其他模块或数据集，可使用 `--module` 与 `--dataset`，比较版本时需固定模型、检索配置和裁判：

```powershell
python evaluation/evaluate.py --module rag.py --dataset evaluation/my_eval.jsonl --split test --mode both --judge
```

注意 `--module` 的路径：`rag.py` 现在依赖同目录下的 `rag_core/` 包，所以**指向项目之外的一份 rag.py 副本会 import 失败**。要对比不同版本的实现，请连同 `rag_core/` 一起拷贝，或在 git 分支上切换后再跑。

## 4. 输出内容

默认输出到 `evaluation/reports/run_<UTC时间>/`，已有目录不会被覆盖。

| 文件 | 内容 |
|---|---|
| `report.md` | 便于查看的汇总表 |
| `summary.json` | 配置、文件哈希、各方法和问题类别的统计、评分覆盖数 |
| `details.jsonl` | 每题每种方法的回答、完整上下文、异常、裁判依据与使用量 |

`summary.json` 里的 `retriever_k` 会记下每个检索器的 k：`rerank` 记的是 `RERANK_TOP_N`（最终条数），`hybrid` 是 `null`（`EnsembleRetriever` 本身没有 k，它由两路子检索器的 k 决定）。

`module_sha256` 仍仅哈希 `rag.py`，不能单独代表整个实现。当前脚本另外保存 `source_sha256`（入口、`file_loader.py`、`rag_core/*.py`）、`evaluator_sha256`，以及重排和流程预算配置。旧报告没有这些字段时，不能仅凭入口文件哈希认定实现相同。

指定新的报告目录：

```powershell
python evaluation/evaluate.py --mode retrieval --output evaluation/reports/baseline_v1
```

每题明细、汇总和报告都会保存；`summary.json` 采用临时文件替换，减少中断时留下半个 JSON 的风险。`running` 表示仍在运行或进程被外部直接终止，并不代表评测完成；以 `completed_evaluations` / `planned_evaluations` 核对覆盖数。正常捕获 Ctrl+C 时状态为 `interrupted`。

供应商明确返回余额不足或无资源包（当前识别错误码 `1113`）时，脚本保存当前记录并停止后续样本，状态为 `resource_exhausted`。如果检索已报告该错误，就跳过本题裁判；裁判自身遇到该错误也不重试。普通 HTTP 429 限流不会仅凭状态码被误判成余额不足。

退出码：0 为完成且没有记录错误，1 为存在问题级运行/证据流程/裁判错误，2 为参数或初始化错误，3 为账户资源耗尽，130 为中断。初始化失败时报告目录中只有 `summary.json`。资源恢复后使用新的 `--output` 目录重新运行，原报告保留。

## 5. 指标含义

| 指标 | 定义与限制 |
|---|---|
| `page_hit_at_k` | 前 K 个片段中，是否至少有一个来自标注证据页 |
| `page_recall_at_k` | 命中的不同标注页数 / 全部标注页数；同页多个片段只算一个命中 |
| `mrr_at_k` | 第一个标注页命中片段的排名倒数，K 内未命中为 0；片段不先按页去重 |
| `route_correct` | Agent 的 `need_retrieve` 是否符合题目标签；不是整个 Agent 的成功率 |
| `route_action_correct` | Agent 的 retrieve/direct/clarify 与无歧义标签是否一致；分母以当次数据和成功执行情况为准 |
| `all_context_page_hit` / `all_context_page_recall` | 全部最终上下文的页命中与标注页覆盖率，单独报告以区别前 K 个片段的指标 |
| `workflow_error_count` | 检索或证据检查异常后返回降级说明的题数；程序执行完成不代表证据检查成功 |
| `citation_present` | 可回答且应检索的问题中，答案是否含 `[资料 N]` 格式引用 |
| `citation_id_validity` | 已出现的引用编号中，有效编号的比例；没有引用时为 null |
| `judge_answer_correct` | LLM 裁判的答案正确率；仅统计成功评分的题目 |
| `judge_grounded` | LLM 裁判判断回答事实是否受实际最终上下文支持；不是 Ragas 的声明级 Faithfulness 指标 |
| `judge_citations_supported` | LLM 裁判判断编号对应的上下文是否支持相关结论；无引用为 null |
| `unanswerable_abstention_rate` | 已评分的不可回答题中，明确无法作答的比例 |
| `answerable_false_abstention_rate` | 已评分的可回答题中，错误拒答的比例 |
| P50/P95 延迟 | 成功请求的延迟分位数；不包含模块初始化和裁判调用 |
| `error_rate` | 方法运行失败题数 / 尝试题数 |

所有均值同时保存 `n`。没有适用样本的指标是 null，不能当作 0。带证据页的执行失败在操作性检索指标中计作未命中；回答质量指标仍只统计已成功评分题，必须结合失败率和覆盖数解释。

**页级标注属于粗粒度评测。** 命中了正确页上的无关片段也可能得分，不能把页级 Recall 当作精确证据召回。若其他页面同样能回答却未被标注，也可能低估效果。应持续补全合理证据页，正式实验可升级到原文证据范围或 chunk 标注。

对于 `agent`，页指标对应**最终保留的上下文**，拒答/澄清时清空；不等同于首次召回质量，初始检索池和节点行为可结合 trace 查看。四路纯检索方法对应直接召回结果。证据覆盖是模型自检诊断，不能当作人工标注的正确率。

每轮重排保留至多 `RERANK_TOP_N` 个子块，Agent 随后扩展父块，并可能合并补搜证据，因此最终上下文数量还受证据条数与 token 预算约束。比较旧版报告时需核对这些配置。`rerank` 延迟包含重排开销，不能与 `hybrid` 当作相同计算工作量比较。

Token 由 LangChain 回调中实际返回的数据统计；缺失时为 null。`reported_total_tokens` 可能仅覆盖部分已返回用量的调用，应同时看 `token_reported_calls`。不包含 Embedding、初始化、以及供应商内部重试的完整成本，也不换算成费用。裁判用量见单独的 `judge_usage`。

## 6. 扩充评测集

JSONL 每行一道题，例如：

```json
{"id":"dev-001","split":"dev","category":"comparison","question":"问题正文","answerable":true,"expected_need_retrieve":true,"reference_answer":"人工核验的参考答案和关键要求","reference_pages":[{"source":"knowledge_db/book/d2l-zh-pytorch.pdf","page_number":168}]}
```

`source` 是相对于项目根目录的文件路径，不能只写书名。`page_number` 是 PDF 从 1 开始的实际页序号，不是书上印刷的页码。没有答案的题目用 `answerable=false`、空证据列表，并在参考答案中写明预期的拒答行为。闲聊可设 `answerable=true`、`expected_need_retrieve=false` 和空证据列表。

`expected_need_retrieve` 是你的产品策略标签：例如缺少用户私有训练信息时，允许系统直接澄清，还是要求先检索，都应事先确定并一致标注。

建议建立开发集和固定测试集，将同一问题的改写、同一组强相关例题放入同一个集合。需要追踪跨书问题、无答案问题以及证据噪声问题，避免只测容易的事实查询。

## 7. 示例题依据

示例题中的页码和参考答案已通过文本提取及页面图像检查：

- `knowledge_db/book/d2l-zh-pytorch.pdf`：PDF 167、168 页（印刷页 149、150）。
- `knowledge_db/book/v1.pdf`：PDF 18、19 页。本地文件是《动手学大模型应用开发》，不是 Happy-LLM。
- 信息缺失题询问用户自己的训练日志，不把教程中的推荐参数当作用户实际配置。

该样例未覆盖《南瓜书》和 Easy-RL，未声称是独立留出测试集。

## 8. 离线检查

```powershell
python -m unittest discover -s evaluation -p "test_*.py" -v
```

这些测试使用假的检索器/Agent 和临时文件检查指标、故障记录与报告流程，不导入真实 `rag.py`，不调用模型。

被测实现自己的离线单测在根目录 `tests/`，覆盖三路路由、整组证据检查、补搜与停止条件、解析失败处理，以及切分、重排、父块扩展和索引缓存。测试使用假模型与临时数据，不调用付费接口；用例数以当前测试输出为准。

```powershell
python -m unittest discover -s tests -t .
```

设计参考：[LangChain 官方 RAG 评测说明](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)。本脚本在本地输出 JSONL/JSON/Markdown，不要求注册 LangSmith，也没有使用 Ragas 的评分实现。
