"""
LLM 接入：统一构造 OpenAI 兼容的 ChatOpenAI（DeepSeek / Ollama / 百炼 通用）。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from kb_core.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    """构造聊天模型；未配置 API Key 时抛出带指引的异常。"""
    if not settings.llm_api_key or settings.llm_api_key.startswith("sk-你的"):
        raise RuntimeError(
            "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填入密钥，"
            "或在 .env 中使用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 指定模型。"
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_retries=2,
        timeout=60,
    )


def build_rewrite_llm(settings: Settings) -> ChatOpenAI:
    """用于查询改写/闲聊的小模型，与主模型共用配置即可。"""
    return build_llm(settings)
