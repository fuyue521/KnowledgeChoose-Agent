"""
向量存储：轻量、零依赖的余弦相似度向量库。

把每个切片的向量与元数据持久化到 runtime/vectors.json，
检索时做余弦相似度排序。每个切片带稳定 chunk_id（写入元数据），
供混合检索按 id 对齐双路召回，避免内容重复导致错配。
数据量在数万条以内足够快，且不依赖外部向量数据库服务。
"""
from __future__ import annotations

import json
import math
import threading
from typing import List, Optional

from langchain_core.documents import Document

from kb_core.config import Settings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """完整余弦相似度（不假定向量已归一化）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class VectorStore:
    """Document 级向量库：add / delete_all / count / search / get_by_id。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._vectors: List[List[float]] = []
        self._documents: List[Document] = []
        self._doc_ids: List[str] = []
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        path = self.settings.vector_store_file
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        self._vectors = [entry["vector"] for entry in data]
        self._documents = [
            Document(page_content=entry["page_content"], metadata=dict(entry.get("metadata", {})))
            for entry in data
        ]
        self._doc_ids = [entry["id"] for entry in data]

    def save(self) -> None:
        self.settings.ensure_dirs()
        data = [
            {
                "id": self._doc_ids[i],
                "vector": self._vectors[i],
                "page_content": self._documents[i].page_content,
                "metadata": self._documents[i].metadata,
            }
            for i in range(len(self._doc_ids))
        ]
        with open(self.settings.vector_store_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ---------- 写 ----------
    def add_documents(self, documents: List[Document], embeddings: List[List[float]], ids: Optional[List[str]] = None) -> None:
        """入库；为每个文档注入稳定 chunk_id（若原 metadata 未带）。"""
        if ids is None:
            ids = [f"chunk-{i}" for i in range(len(documents))]
        with self._lock:
            for doc, vec, cid in zip(documents, embeddings, ids):
                md = dict(doc.metadata)
                md.setdefault("chunk_id", cid)
                self._vectors.append(list(vec))
                self._documents.append(Document(page_content=doc.page_content, metadata=md))
                self._doc_ids.append(cid)
        self.save()

    def delete_all(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._documents.clear()
            self._doc_ids.clear()
        self.save()

    @property
    def count(self) -> int:
        return len(self._doc_ids)

    def all_documents(self) -> List[Document]:
        return list(self._documents)

    def chunk_id_at(self, index: int) -> str:
        return self._doc_ids[index]

    # ---------- 读 ----------
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Document]:
        """返回相似度降序的 top_k 文档（附 score 元数据）。"""
        scored = [
            (i, cosine_similarity(query_vector, self._vectors[i]))
            for i in range(len(self._vectors))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[Document] = []
        for i, score in scored[:top_k]:
            doc = Document(
                page_content=self._documents[i].page_content,
                metadata=dict(self._documents[i].metadata),
            )
            doc.metadata["vector_score"] = round(score, 6)
            results.append(doc)
        return results
