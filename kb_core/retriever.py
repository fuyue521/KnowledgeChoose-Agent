"""
混合检索器：向量召回 + BM25 召回，用 RRF（Reciprocal Rank Fusion）融合排序。
RRF 对两路排名的名次做倒数加权，鲁棒性优于直接相加分数。

两路召回通过每个切片的稳定 chunk_id 对齐，检索结果按 chunk_id 去重，
避免内容相同的切片被重复计入或错配。
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document

from kb_core.bm25 import BM25Index
from kb_core.config import Settings
from kb_core.store import VectorStore

_RRF_K = 60.0


class HybridRetriever:
    def __init__(self, settings: Settings, vector_store: VectorStore, bm25: BM25Index):
        self.settings = settings
        self.vector_store = vector_store
        self.bm25 = bm25
        # 索引与 chunk_id 的映射，供 BM25（按下标召回）转 chunk_id
        self._id_by_index: Dict[int, str] = {}

    def refresh(self) -> None:
        """向量库重建后刷新 chunk_id 映射。"""
        self._id_by_index = {}
        for i in range(self.vector_store.count):
            self._id_by_index[i] = self.vector_store.chunk_id_at(i)

    def search(self, query: str, query_vector: List[float], top_k: int | None = None) -> List[Document]:
        """融合检索，返回带 score 元数据的文档列表（已按 chunk_id 去重）。"""
        top_k = top_k or self.settings.final_top_k
        n = self.vector_store.count
        if n == 0:
            return []
        if len(self._id_by_index) != n:
            self.refresh()

        # 1) 向量路：前 vector_top_k 名次
        vec_docs = self.vector_store.search(query_vector, top_k=self.settings.vector_top_k)
        # 2) BM25 路：按下标召回前 bm25_top_k
        bm25_hits = self.bm25.search(query, top_k=self.settings.bm25_top_k)

        rrf: Dict[str, float] = {}  # chunk_id -> rrf score
        for rank, doc in enumerate(vec_docs):
            cid = doc.metadata.get("chunk_id")
            if cid:
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, (idx, _score) in enumerate(bm25_hits):
            cid = self._id_by_index.get(idx)
            if cid:
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # 取回命中文档原文
        doc_by_id: Dict[str, Document] = {d.metadata.get("chunk_id"): d for d in self.vector_store.all_documents()}
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[Document] = []
        for cid, score in merged:
            doc = doc_by_id.get(cid)
            if doc is None:
                continue
            out = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            out.metadata["retrieval_score"] = round(score, 6)
            results.append(out)
        return results
