# 四本书评测集

开发集 **120 题**，测试集 **200 题**，共 **320 题**。原扩充版已经包含早期全部题目，因此统一目录时保留完整集，不重复拼接。题目、参考答案、证据、题号及 dev/test 划分均保持不变。

## 文件与用途

| 文件 | 用途 |
|---|---|
| [dev.jsonl](dev.jsonl) | 开发集，用于调参、排查问题和回归 |
| [test.jsonl](test.jsonl) | 测试集，用于配置冻结后的检验 |
| [manifest.json](manifest.json) | 类别统计、源 PDF 与数据文件的 SHA-256、证据提取方式 |
| [topic_catalog.json](topic_catalog.json) | 主题名称、所属 split 和对应题号 |
| [validation_report.json](validation_report.json) | 离线数据核验记录，不是模型效果报告 |

JSONL 每行是一道题。直接维护这两份数据文件，不再保留重复的增量文件、逐题 Markdown 副本或依赖旧版本目录的生成脚本。修改数据后需同步核对主题目录、统计和哈希；旧校验记录仅适用于记录中的数据哈希。

## 数量与覆盖

| 类别 | dev | test | 合计 |
|---|---:|---:|---:|
| `fact`：概念、代码或规则理解 | 36 | 67 | 103 |
| `paraphrase`：同义改写 | 16 | 24 | 40 |
| `multi_evidence`：多要点或综合推理 | 28 | 51 | 79 |
| `cross_book`：跨书综合 | 12 | 16 | 28 |
| `unanswerable`：事实材料缺失 | 16 | 22 | 38 |
| `clarification`：目标或指代需澄清 | 8 | 10 | 18 |
| `direct`：可直接完成的请求 | 4 | 10 | 14 |
| **总计** | **120** | **200** | **320** |

250 题带知识库证据，覆盖 90 个知识主题标签、160 个不同的证据页。多个问题可以共用主题，跨书题也会连接主题；320 道题不等于 320 个独立知识点。

`diagnostic_tags` 只在部分题目中标注，不能用其数量代表整套数据的完整分类。`multi_evidence` 不强制跨页；`adjacent_page_evidence` 表示证据涉及相邻 PDF 页，不保证跨过某种切分器的实际 chunk 边界。

## 每条数据的字段

| 字段 | 含义 |
|---|---|
| `id` | 稳定题号；保留部分题号原有的 `v2-` 前缀，方便对照历史报告，不再对应版本目录 |
| `split` | `dev` 或 `test` |
| `category` | 主类别，用于分类统计 |
| `topic_ids` | 知识或行为主题；名称见主题目录 |
| `question` | 输入被测系统的用户问题 |
| `answerable` | 当前信息是否足以完整作答；澄清题为 false 不代表系统失败 |
| `expected_need_retrieve` | 兼容二值路由的标签；无答案题有特殊约定，见下文 |
| `expected_route` | 部分题目的明确三路动作：`retrieve`、`clarify`、`direct` |
| `reference_answer` | 参考答案，允许合理等价表达及证据支持的推导 |
| `reference_pages` | PDF 文件及从 1 开始的物理页序号，不是书内印刷页码 |
| `reference_evidence` | 证据原文、页号、提取方式和规范化文本中的字符区间 |
| `rubric.required_points` | 答案应覆盖的评分要点；现有评测器尚未逐点计分 |
| `expected_behavior` | 应基于证据回答、追问、直接回答，还是说明信息不足 |
| `diagnostic_tags` | 可选的附加诊断标签；现有评测器未按它自动分组 |
| `acceptable_routes` | 可接受的路由集合；现有评测器尚未据此计分 |
| `missing_information` | 为完整作答还需要用户提供的信息 |
| `annotation_status` / `annotation_note` | 标注来源和限制，不代表已完成独立人工审核 |

错误前提题可以是 `answerable=true`：例如“保存 state_dict 就保存了任意模型架构吗？”有依据可纠正，应给出纠正答案。缺少实际权重、用户日志或运行结果的题目，则不能用教材内容编造具体事实。

### 证据坐标

先提取整页文本，做 Unicode NFKC 规范化，再移除所有空白字符。`[start_char, end_char)` 是该文本的左闭右开区间，不是 PDF 坐标、原始文本偏移或 chunk ID。

- 未指定 `extractor` 的证据使用 `pypdfium2` 的 `get_text_range()`。
- 指定 `extractor="pymupdf"` 的证据使用 PyMuPDF 的 `page.get_text()`。

不同提取器对公式、连字符和文本顺序的处理可能不同，不能混用字符偏移。合并前已重新离线核验全部 **377 处证据**；合并后的 JSONL 与核验时文件逐字节一致。源文本匹配不证明参考答案完全正确，也不代替独立人工复核。

### 路由标签

无答案题的 `expected_need_retrieve=true` 沿用“检索后判断证据不足”的兼容约定。但明确缺少材料时，直接说明不足或追问也合理；部分此类题目列出多个 `acceptable_routes`，不指定唯一 `expected_route`。

当前三路 `route_action_correct` 默认排除没有明确路由的无答案题；二值 `route_correct` 仍使用兼容标签。分析时需结合类别和实际行为，不能仅用二值路由得分判断系统好坏。

## 使用方法

在项目根目录运行。**不传 `--dataset` 时，评测器仍默认使用 6 题 smoke 集。**

只检查格式、标签及来源路径，不调用模型：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --validate-only
python evaluation/evaluate.py --dataset evaluation/datasets/test.jsonl --validate-only
```

此命令不解析 PDF，不核验页码上限、证据原文或答案内容；已有的完整证据核验结果记录在 `validation_report.json`。

运行开发集 Agent 评测，会调用配置的问答及裁判接口：

```powershell
python evaluation/evaluate.py --dataset evaluation/datasets/dev.jsonl --mode agent --judge --judge-model deepseek-flash --warmup-rerank
```

配置冻结后，将路径改为 `evaluation/datasets/test.jsonl` 可运行测试集。其他选项和指标解释见 [评测说明](../README.md)。

## 开发与测试的边界

- 用 dev 调整切分、召回、重排、路由和 Prompt，测试前冻结配置与 Judge 设置。
- 两组的标注主题和证据页无交叉，但更广泛的概念仍可能相关，这不是跨领域泛化测试。
- 如果已根据某些 test 题的结果调参，就不能再把这些题称为未见过的盲测题。合并目录不会恢复测试独立性。
- 题目按教材构造，不代表真实用户问题分布；参考页也未穷尽所有合理证据来源。独立人工复核尚未完成。
- 评测数据不得放入 `knowledge_db/`。本次只是整理文件，不需要重新入库。

指标缺口与后续建议见 [评测审查说明](../../docs/evaluation_audit_20261022.md)。
