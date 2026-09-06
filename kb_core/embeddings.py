"""
Embedding 向量化抽象：
- openai    : 任意 OpenAI 兼容 embedding 接口（Ollama / 百炼 / 硅基流动等）
- huggingface: 本地 bge 系列（可选依赖）
- debug     : 免密钥的确定性哈希向量，仅供离线冒烟测试 / 无网络演示
统一实现 langchain_core.embeddings.Embeddings 接口。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

from kb_core.config import Settings


class DebugEmbeddings(Embeddings):
    """免密钥 / 免模型的确定性向量。

    将文本按字符 n-gram 做哈希投影到固定维度并归一化。
    效果远弱于真实语义模型，只用于把整条 RAG 流程在没有网络时跑通。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vectorize(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        # 提取中英文/数字 token，再取 1~2 元字符组
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        grams = set()
        for tok in tokens:
            if len(tok) <= 2:
                grams.add(tok)
            for i in range(len(tok)):
                grams.add(tok[i])
                if i + 1 < len(tok):
                    grams.add(tok[i : i + 2])
        for g in grams:
            d = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(d[:4], "little") % self.dim
            sign = 1.0 if d[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)


def build_embeddings(settings: Settings) -> Embeddings:
    """按配置构造 embedding 模型。"""
    provider = settings.embedding_provider.lower()
    if provider == "debug":
        return DebugEmbeddings(settings.debug_embed_dim)

    if provider in ("openai", "ollama", "openai_compatible"):
        base_url = settings.embedding_base_url or "http://localhost:11434/v1"
        api_key = settings.embedding_api_key or "ollama"
        model = settings.embedding_model or "nomic-embed-text"
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("缺少 langchain-openai，请先 pip install langchain-openai") from e
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            chunk_size=settings.embedding_batch_size,
            check_embedding_ctx_length=False,
        )

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "缺少 sentence-transformers / langchain-huggingface，"
                "请先：pip install sentence-transformers langchain-huggingface"
            ) from e
        model = settings.embedding_model or "BAAI/bge-small-zh-v1.5"
        return HuggingFaceEmbeddings(model_name=model)

    raise ValueError(
        f"未知 embedding_provider: {settings.embedding_provider}（可选 openai | huggingface | debug）"
    )
