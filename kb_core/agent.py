"""
LangChain Agent 编排（langchain 1.x 的 create_agent）。

工具集：
1. knowledge_base_query —— RAG 知识库检索问答（商品/政策/售后等）
2. check_order         —— 模拟订单/物流查询

Agent 自行判断该调用哪个工具、是否需要继续追问、何时直接回复。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool

from kb_core.config import Settings
from kb_core.history import SessionMemory
from kb_core.llm import build_llm
from kb_core.order_tool import to_tool_function as order_tool_fn
from kb_core.rag import KnowledgeBase

logger = logging.getLogger(__name__)

_AGENT_SYSTEM_PROMPT = """你是「知选商城」的智能客服 Agent，负责在电商场景下帮助用户解决问题。

## 你的工具
1. knowledge_base_query：知识库问答。适合查询商品规格、退换货政策、物流规则、发票、会员/优惠券、售后保修等知识库内容。
2. check_order：订单查询。当用户询问具体订单状态、物流进度、是否发货时使用。

## 工作准则
1. 先判断意图：闲聊/问候直接友好回复；知识性问题用 knowledge_base_query；订单问题用 check_order。
2. 调用知识库工具后，把工具返回的内容组织成通顺、有礼貌的客服回答；不要展示来源文件名、
   参考文档或【来源】标记。
3. 工具返回“超出知识范围”之类的说明时，如实转达并引导联系人工客服。
4. 用户没说清订单号时，引导用户提供 XZ 开头的订单号。
5. 回答使用中文，简洁自然，不要输出思考过程，不要把“工具调用”讲给用户听。"""


def _strip_source_markers(answer: str) -> str:
    """兜底清理模型偶发输出的来源行，避免出现在 Agent 回答中。"""
    markers = ("【来源】", "来源：", "来源:", "参考文档：", "参考文档:", "📎 参考文档")
    lines = answer.splitlines()
    kept = [line for line in lines if not line.strip().startswith(markers)]
    return "\n".join(kept).strip()


class EcommerceAgent:
    def __init__(
        self,
        settings: Settings,
        knowledge_base: KnowledgeBase,
        memory: SessionMemory,
        llm=None,
    ):
        self.settings = settings
        self.knowledge_base = knowledge_base
        self.memory = memory
        self._agent = None
        self._llm = llm  # 复用外部传入模型实例；为空则懒加载

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    def _build_agent(self):
        if self._agent is not None:
            return self._agent
        tools: List[Tool] = [
            Tool(
                name="knowledge_base_query",
                func=self.knowledge_base.to_tool_function(),
                description=(
                    "查询「知选商城」知识库：商品规格、退换货、物流、发票、会员优惠、售后保修等。"
                    "输入用户问题，返回基于知识库的客服回答。当用户问店铺政策或商品信息时必须使用它。"
                ),
            ),
            Tool(
                name="check_order",
                func=order_tool_fn(self.settings),
                description=(
                    "查询订单状态与物流信息。当用户询问自己的订单/物流/发货情况时使用。"
                    "输入用户的整句话，工具会自动提取 XZ 开头的订单号。"
                ),
            ),
        ]
        try:
            self._agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=_AGENT_SYSTEM_PROMPT,
                name="zhixuan_cs_agent",
            )
        except Exception as e:  # 模型不支持工具调用等情况
            logger.warning("create_agent 失败，回退到纯知识库问答: %s", e)
            raise RuntimeError(
                f"创建 Agent 失败（{e}）。若您的模型不支持 function calling，请在界面选择「chain 模式」。"
            ) from e
        return self._agent

    def chat(self, question: str, session_id: str) -> str:
        """带会话历史地调用 Agent。"""
        agent = self._build_agent()
        from langchain_core.messages import AIMessage, HumanMessage

        messages = []
        for q, a in self.memory.get(session_id):
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        messages.append(HumanMessage(content=question))

        resp = agent.invoke({"messages": messages})
        # langgraph 输出：取最后一则 AI 消息
        last = resp.get("messages", [])[-1] if isinstance(resp, dict) else None
        answer = getattr(last, "content", None) or str(resp)
        return _strip_source_markers(str(answer))
