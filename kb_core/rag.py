"""
知识库 RAG 工具：检索 + 生成一站式工具。
供 Agent 作为工具调用，也可直接用于 chain 模式。

返回结构化的（answer, sources, docs），由上层展示。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from kb_core.config import Settings
from kb_core.llm import build_llm
from kb_core.retriever import HybridRetriever
from kb_core.store import VectorStore

_SYSTEM_TEMPLATE = """你是一家名为「知选商城」的专业电商客服，负责处理知识库问题和当前会话问题。

规则：
1. 商品、价格、政策、物流等事实问题必须以【知识库资料】为准，不得编造资料中没有的内容。
2. 如果用户询问当前对话、之前的问题或之前的回答，优先依据【当前会话历史】回答，不要把它强行当成知识库问题。
3. 回答要简洁、口语化、有温度，像真人客服。
4. 若知识库资料和会话历史都不足以回答，明确告诉用户“这个问题超出了我的知识范围，建议联系人工客服 400-888-0000”。
5. 若资料中提到多个条件，请完整说明，不要遗漏例外情况。
6. 在使用知识库资料回答时，末尾换行列出引用来源，格式为：【来源】文件名；纯会话问题不需要伪造来源。
7. 不要输出“根据资料”之类的机器腔。

【当前会话历史】
{history}

【知识库资料】
{context}"""


@dataclass
class RagResult:
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    docs: List[Document] = field(default_factory=list)
    hit: bool = False


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        score = doc.metadata.get("retrieval_score", doc.metadata.get("vector_score", ""))
        score_txt = f"（相关度 {score}）" if score != "" else ""
        blocks.append(f"[{i}] 来源：{source}{score_txt}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _format_history(history: Sequence[Tuple[str, str]] | None) -> str:
    if not history:
        return "（暂无历史对话）"
    lines = []
    for question, answer in history:
        lines.append(f"用户：{question}")
        lines.append(f"客服：{answer}")
    return "\n".join(lines)


class KnowledgeBase:
    """把向量库 + BM25 + LLM 合成封装成一个可被 Agent 调用的知识库。"""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        embeddings: Embeddings | None = None,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.retriever = retriever
        self._embeddings = embeddings
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def embeddings(self) -> Embeddings:
        """复用同一个 embedding 实例，避免每次提问重建模型。"""
        if self._embeddings is None:
            from kb_core.embeddings import build_embeddings

            self._embeddings = build_embeddings(self.settings)
        return self._embeddings

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        history: Sequence[Tuple[str, str]] | None = None,
        retrieval_query: str | None = None,
    ) -> RagResult:
        """完整 RAG：检索 -> 生成。"""
        result = RagResult()
        if self.vector_store.count == 0:
            result.answer = "知识库还没有内容，请先运行 python build_index.py 构建索引。"
            return result

        search_query = retrieval_query or question
        query_vec = self.embeddings.embed_query(search_query)
        docs = self.retriever.search(search_query, query_vec, top_k=top_k or self.settings.final_top_k)
        if not docs and not history:
            result.answer = "没有检索到相关内容，建议换个问法或联系人工客服。"
            return result

        result.docs = docs
        result.hit = True
        # 去重整理来源
        seen = set()
        for d in docs:
            s = d.metadata.get("source", "未知")
            if s not in seen:
                seen.add(s)
                result.sources.append(s)

        context = _format_context(docs) if docs else "（本次没有检索到相关知识库资料）"
        messages = [
            SystemMessage(
                content=_SYSTEM_TEMPLATE.format(
                    history=_format_history(history),
                    context=context,
                )
            ),
            HumanMessage(content=question),
        ]
        try:
            answer = self.llm.invoke(messages).content
            result.answer = (answer or "").strip()
        except Exception as e:
            # Web/CLI 兜底：LLM 不可用时仍返回一份可读的检索结果，避免整个界面报错。
            result.answer = self._fallback_answer(question, docs)
            result.answer += f"\n\n（提示：模型调用失败，已返回检索兜底结果：{type(e).__name__}）"
        return result

    def _fallback_answer(self, question: str, docs: List[Document]) -> str:
        """无 LLM 时的保底回答。"""
        if not docs:
            return "当前没有检索到相关知识库资料，模型也暂时无法根据会话历史生成回答。"
        lines = ["我先帮你查到这些相关资料："]
        for i, doc in enumerate(docs[:3], 1):
            source = doc.metadata.get("source", "未知来源")
            text = " ".join(doc.page_content.split())
            if len(text) > 140:
                text = text[:140].rstrip() + "..."
            lines.append(f"{i}. {source}：{text}")
        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "未知")
            if source not in seen:
                seen.add(source)
                sources.append(source)
        if sources:
            lines.append("")
            lines.append("来源：" + "、".join(sources))
        lines.append("如果你愿意，我也可以继续帮你缩小到更具体的问题。")
        return "\n".join(lines)

    def to_tool_function(self):
        """Agent 工具函数：入参为查询字符串，返回基于知识库的客服回答文本。
        注意：Agent 层自行组织话术与是否展示来源，工具只负责给出可靠答案。"""

        def search(query: str) -> str:
            res = self.ask(query)
            return res.answer

        search.__name__ = "knowledge_base_query"
        search.__doc__ = (
            "查询「知选商城」的商品规格、退换货、物流、发票、会员优惠、售后保修等知识库内容。"
            "输入用户想了解的问题，返回基于知识库的客服回答。"
        )
        return search
