"""rag_core 的离线单元测试。

不 import rag.py（那会真的建索引、真的调接口），直接测 rag_core 里的纯逻辑。
一次网络请求都不发，一秒出头就跑完。

重排那几个模块也遵守同一条纪律：`rag_core/rerank.py` 在模块级不 import torch，
测试里用的都是假的 scorer，所以测 sigmoid 和排序不必加载 1.1GB 权重。

在项目根目录运行：

    python -m unittest discover -s tests -t . -v
"""