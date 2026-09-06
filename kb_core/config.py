"""
全局配置：从环境变量 / .env 文件读取，提供统一配置对象。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = kb_core 的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 独立项目只读取自身根目录的 .env，不依赖父目录配置。
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


@dataclass
class Settings:
    """集中管理所有可配置项，构造后可通过字段访问。"""

    # ----- 目录 -----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    orders_file: Path = PROJECT_ROOT / "data" / "orders_sample.json"
    runtime_dir: Path = PROJECT_ROOT / "runtime"
    vector_store_file: Path = PROJECT_ROOT / "runtime" / "vectors.json"

    # ----- LLM -----
    # LLM 配置只来自当前项目的 .env 或显式环境变量。
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))

    # ----- Embedding -----
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "openai"))
    embedding_base_url: str = field(default_factory=lambda: _env("EMBEDDING_BASE_URL"))
    embedding_api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL"))
    debug_embed_dim: int = 256
    # 百炼 Qwen Embedding 等接口单次批量上限常见为 20
    embedding_batch_size: int = field(default_factory=lambda: int(_env("EMBEDDING_BATCH_SIZE", "20")))

    # ----- 切片 -----
    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "80")))

    # ----- 混合检索 -----
    vector_top_k: int = field(default_factory=lambda: int(_env("RAG_VECTOR_TOP_K", "8")))
    bm25_top_k: int = field(default_factory=lambda: int(_env("RAG_BM25_TOP_K", "8")))
    final_top_k: int = field(default_factory=lambda: int(_env("RAG_FINAL_TOP_K", "5")))

    # ----- 问答模式 -----
    default_mode: str = field(default_factory=lambda: _env("DEFAULT_MODE", "agent"))  # agent | chain

    # ----- 记忆 -----
    max_session_turns: int = 12

    def ensure_dirs(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
