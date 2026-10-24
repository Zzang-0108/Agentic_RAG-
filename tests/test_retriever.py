"""HNSWRetriever：用真 faiss 索引 + 假 embedding 跑，不联网、不花钱。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402

from rag_core.indexing import HNSWRetriever, build_faiss_index  # noqa: E402


class FixedEmbeddings:
    """按文本返回写死的向量，用来构造可预期的相似度顺序。"""

    def __init__(self, table):
        self.table = table
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.table[text]


DOCS = [
    Document(page_content="甲", metadata={"source": "a.pdf", "page_number": 1}),
    Document(page_content="乙", metadata={"source": "b.pdf", "page_number": 2}),
    Document(page_content="丙", metadata={"source": "c.pdf", "page_number": 3}),
]

# 三个几乎正交的方向 + 一个正好等于「甲」的查询向量
VECTORS = {
    "甲": [1.0, 0.0, 0.0],
    "乙": [0.0, 1.0, 0.0],
    "丙": [0.0, 0.0, 1.0],
    "查询甲": [1.0, 0.0, 0.0],
}


def make_retriever(k=2, docs=None):
    docs = list(docs if docs is not None else DOCS)
    import numpy as np

    matrix = np.array([VECTORS[doc.page_content] for doc in docs], dtype=np.float32)
    embeddings = FixedEmbeddings(VECTORS)
    index = build_faiss_index(matrix)  # 注意：build_faiss_index 是原地归一化已传进去的矩阵
    return HNSWRetriever(index=index, documents=docs, embeddings=embeddings, k=k), embeddings, index


class TestHNSWRetriever(unittest.TestCase):
    def test_returns_k_documents_ranked_by_similarity(self):
        retriever, _, _ = make_retriever(k=2)
        out = retriever.invoke("查询甲")

        self.assertEqual(len(out), 2)
        self.assertIs(out[0], DOCS[0])  # 和查询向量完全相同的那一篇排第一
        self.assertEqual(
            [doc.metadata["score"] for doc in out],
            sorted([doc.metadata["score"] for doc in out], reverse=True),
        )
        self.assertAlmostEqual(out[0].metadata["score"], 1.0, places=5)
        self.assertAlmostEqual(out[1].metadata["score"], 0.0, places=5)

    def test_index_holds_every_chunk(self):
        _, _, index = make_retriever(k=1)
        self.assertEqual(index.ntotal, len(DOCS))
        self.assertEqual(index.hnsw.efSearch, 64)
        self.assertEqual(index.hnsw.efConstruction, 200)

    def test_query_goes_through_embed_query(self):
        retriever, embeddings, _ = make_retriever(k=1)
        retriever.invoke("查询甲")
        self.assertEqual(embeddings.queries, ["查询甲"])

    def test_score_is_written_onto_the_shared_document(self):
        """HNSWRetriever 里的 documents 就是外面那份 splits，score 是写回去的（有副作用）。"""
        retriever, _, _ = make_retriever(k=1)
        DOCS[0].metadata.pop("score", None)  # 别的用例可能已经写过，先清掉再验
        retriever.invoke("查询甲")
        self.assertIn("score", DOCS[0].metadata)

    def test_k_larger_than_index_still_terminates(self):
        """k 比索引里的文档还多时，faiss 会在末尾补 -1，跳过即可，不能崩。

        顺带钉住那个已知的下标错位问题（`scores[0][len(result)]`）：
        实测 faiss 的 -1 只出现在**末尾**，所以当前这种用法下拿不到重复分数；
        一旦哪天 -1 出现在中间，分数就会整体错位。行为按原样保留，不改。
        """
        retriever, _, _ = make_retriever(k=5)
        out = retriever.invoke("查询甲")

        self.assertEqual(len(out), len(DOCS))
        scores = [doc.metadata["score"] for doc in out]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(score > -1e30 for score in scores))  # 没把 -inf 那几档抄进来


if __name__ == "__main__":
    unittest.main()