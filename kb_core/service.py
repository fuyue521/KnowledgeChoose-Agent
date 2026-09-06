"""
门面服务层：把配置、索引构建、检索、Agent、记忆统一封装，
供 Web UI / CLI / 评测脚本共用同一个入口。

用法：
    svc = AssistantService()
    svc.build_index()                 # 一次性构建知识库索引
    answer = svc.ask("你们怎么退货？", session_id="u1", mode="agent")
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from kb_core.agent import EcommerceAgent
from kb_core.bm25 import BM25Index
from kb_core.config import Settings
from kb_core.embeddings import build_embeddings
from kb_core.history import SessionMemory
from kb_core.loader import load_raw_documents
from kb_core.rag import KnowledgeBase
from kb_core.retriever import HybridRetriever
from kb_core.store import VectorStore

logger = logging.getLogger(__name__)

# 查询改写：多轮对话里把省略上下文的问题补全（chain 模式专用）
_REWRITE_TEMPLATE = """下面是一段用户与电商客服的对话历史与用户最新的提问。
请把最新提问改写成一个能脱离对话历史、独立检索知识库的完整问句。
只输出改写后的问句，不要输出其它内容。

对话历史：
{history}

用户最新提问：{question}

改写后的问句："""


class AssistantService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.embeddings = build_embeddings(self.settings)
        self.vector_store = VectorStore(self.settings)
        self.memory = SessionMemory(max_turns=self.settings.max_session_turns)

        # 索引相关组件在 build_index / 首次检索时构建
        self._bm25: BM25Index | None = None
        self._retriever: HybridRetriever | None = None
        self._kb: KnowledgeBase | None = None
        self._agent: EcommerceAgent | None = None
        self._llm = None

        # 加载已有索引
        if self.vector_store.count > 0:
            self._rebuild_runtime_index()

    # ---------------- 索引 ----------------
    def _rebuild_runtime_index(self):
        docs = self.vector_store.all_documents()
        self._bm25 = BM25Index([d.page_content for d in docs])
        self._retriever = HybridRetriever(self.settings, self.vector_store, self._bm25)
        self._retriever.refresh()

    def build_index(self, verbose: bool = True) -> int:
        """重建索引：清空 -> 重新切片 -> 向量化 -> 建 BM25。返回切片数。"""
        docs = load_raw_documents(self.settings)
        if not docs:
            return 0
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        if verbose:
            logger.info("切片数量：%d", len(docs))
        self.vector_store.delete_all()
        self.vector_store.add_documents(docs, vectors)
        self._rebuild_runtime_index()
        if verbose:
            logger.info("索引构建完成，共 %d 个切片", self.vector_store.count)
        return len(docs)

    # ---------------- 懒加载 LLM / KB / Agent ----------------
    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            if self.vector_store.count == 0:
                raise RuntimeError("知识库为空，请先运行：python build_index.py")
            self._kb = KnowledgeBase(self.settings, self.vector_store, self.retriever, self.embeddings)
        return self._kb

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            if self.vector_store.count == 0:
                raise RuntimeError("知识库为空，请先运行：python build_index.py")
            self._rebuild_runtime_index()
        return self._retriever

    @property
    def agent(self) -> EcommerceAgent:
        if self._agent is None:
            self._agent = EcommerceAgent(self.settings, self.kb, self.memory, self._llm)
        return self._agent

    # ---------------- 查询改写 ----------------
    def _rewrite_question(self, question: str, session_id: str) -> str:
        history = self.memory.get(session_id)
        if not history:
            return question
        history_text = self.memory.format_prompt(session_id)
        msg = HumanMessage(content=_REWRITE_TEMPLATE.format(history=history_text, question=question))
        try:
            from kb_core.llm import build_llm

            if self._llm is None:
                self._llm = build_llm(self.settings)
            rewritten = self._llm.invoke([msg]).content.strip()
            return rewritten or question
        except Exception as e:  # 未配置 key / 调用失败时用原问题兜底
            logger.warning("查询改写失败，使用原问题：%s", e)
            return question

    # ---------------- 对外主入口 ----------------
    def ask(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "agent",
    ) -> dict:
        """统一问答入口。返回 {"answer", "sources", "mode", "session_id"}。"""
        mode = (mode or self.settings.default_mode or "agent").lower()
        question = (question or "").strip()
        if not question:
            return {
                "answer": "请先告诉我您想咨询什么～",
                "sources": [],
                "docs": [],
                "mode": mode,
                "route": mode,
                "session_id": session_id,
                "fallback": False,
            }

        route = mode
        fallback = False
        if mode == "chain":
            # 改写后的问题只用于检索；最终回答仍使用原问题，并带上会话历史。
            history = self.memory.get(session_id)
            full_q = self._rewrite_question(question, session_id)
            result = self.kb.ask(
                question,
                history=history,
                retrieval_query=full_q,
            )
            answer = result.answer
            sources = result.sources
        elif mode == "agent":
            try:
                answer = self.agent.chat(question, session_id)
            except Exception as e:
                # Agent 不可用（模型不支持工具 / 网络异常等）时退化为 chain 模式
                logger.warning("Agent 模式失败，回退 chain：%s", e)
                history = self.memory.get(session_id)
                full_q = self._rewrite_question(question, session_id)
                result = self.kb.ask(
                    question,
                    history=history,
                    retrieval_query=full_q,
                )
                answer = result.answer
                sources = result.sources
                route = "fallback"
                fallback = True
                # 仅当失败疑似配置问题时，向用户附一句提示
                hint = str(e)
                if ("LLM_API_KEY" in hint) or ("EMBEDDING" in hint) or ("api key" in hint.lower()):
                    answer += "\n\n（提示：Agent 模式异常已自动回退知识库问答，请检查 .env 配置）"
            else:
                sources = []  # Agent 回答已内嵌工具来源
        else:
            raise ValueError(f"未知模式：{mode}（可选 agent / chain）")

        self.memory.add(session_id, question, answer)
        return {
            "answer": answer,
            "sources": sources,
            "docs": getattr(result, "docs", []) if mode != "agent" or fallback else [],
            "mode": mode,
            "route": route,
            "fallback": fallback,
            "session_id": session_id,
        }

    def clear_session(self, session_id: str) -> None:
        self.memory.clear(session_id)
