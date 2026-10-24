# 本地知识库

将 PDF 放入 `knowledge_db/book/`，支持递归读取子目录。PDF 文件不包含在仓库中；文本加载器当前不提供 OCR，请使用能提取文字的版本。

内置开发集和测试集对应以下四个本地文件：

| 文件名 | 资料 |
|---|---|
| `d2l-zh-pytorch.pdf` | 《动手学深度学习》PyTorch 中文版 |
| `EasyRL_v1.0.6.pdf` | Easy RL 强化学习教程 |
| `pumpkin_book.pdf` | 《南瓜书》 |
| `v1.pdf` | 《动手学大模型应用开发》 |

评测使用 PDF 从 1 开始的物理页号和文件哈希。更换 PDF 版本或改名后，需要重新核对评测集的来源与页码；具体哈希见 [数据集 manifest](../evaluation/datasets/manifest.json)。使用自己的其他资料时，也需建立对应问题和证据标注。

在项目根目录运行以下命令准备索引；不调用问答模型或裁判：

```powershell
python -m rag_core.prepare_index --strategy semantic
```

首次运行需要下载本地模型权重。已有匹配索引时会直接加载。切分策略和参数见 [配置说明](../docs/configuration.md)。
