"""
BM25 关键词检索（中文分词优先 jieba，缺失时回退 CJK 二元组切分）。
与向量检索互补：向量擅长语义近似，BM25 擅长精确术语匹配（如型号、政策编号）。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

try:
    import jieba
    jieba.setLogLevel(60)  # 关闭分词日志噪音
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _cjk_bigrams(text: str) -> List[str]:
    """无 jieba 时：对连续中文按单字+相邻二元组切分，保留英文/数字词。"""
    tokens: List[str] = []
    for seg in _WORD_RE.findall(text):
        if _CJK_RE.match(seg):
            if len(seg) == 1:
                tokens.append(seg)
                continue
            # 单字 + 二元组，兼顾精确与召回
            for ch in seg:
                tokens.append(ch)
            for i in range(len(seg) - 1):
                tokens.append(seg[i : i + 2])
        else:
            tokens.append(seg.lower())
    return tokens


def tokenize(text: str) -> List[str]:
    """中文分词；无 jieba 时退化为 CJK 二元组切分。"""
    text = (text or "").lower()
    if _HAS_JIEBA:
        return [t for t in jieba.lcut(text) if t.strip()]
    return [t for t in _cjk_bigrams(text) if t.strip()]


class BM25Index:
    """内存 BM25（Okapi BM25），从知识库全量切片构建。"""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self._docs: List[List[str]] = [tokenize(d) for d in documents]
        self._doc_count = len(self._docs)
        self._avgdl = (
            sum(len(d) for d in self._docs) / self._doc_count if self._doc_count else 0.0
        )
        self._k1 = k1
        self._b = b
        # 文档频率 df：每个词出现在多少篇文档中
        df: Dict[str, int] = {}
        for tokens in self._docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._df = df
        # IDF
        n = self._doc_count
        self._idf: Dict[str, float] = {}
        for term, f in df.items():
            self._idf[term] = math.log(1.0 + (n - f + 0.5) / (f + 0.5))

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """返回 [(doc_index, bm25_score)] 降序列表。"""
        if self._doc_count == 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = [0.0] * self._doc_count
        for term in set(q_tokens):
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, tokens in enumerate(self._docs):
                f = sum(1 for t in tokens if t == term)
                if f == 0:
                    continue
                denom = f + self._k1 * (1.0 - self._b + self._b * (len(tokens) / self._avgdl if self._avgdl else 1.0))
                scores[i] += idf * ((f * (self._k1 + 1.0)) / denom)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked if s > 0][:top_k]
