"""
会话历史记忆：按 session_id 保存多轮对话，自动截断最旧轮次。
"""
from __future__ import annotations

from collections import deque, OrderedDict
from typing import Deque, Dict, List, Tuple


class SessionMemory:
    """内存版会话历史（作业演示足够；生产可换 Redis）。"""

    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._sessions: Dict[str, Deque[Tuple[str, str]]] = OrderedDict()

    def add(self, session_id: str, question: str, answer: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns)
        self._sessions[session_id].append((question, answer))

    def get(self, session_id: str) -> List[Tuple[str, str]]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def format_prompt(self, session_id: str) -> str:
        """把历史拼成文本，供查询改写使用。"""
        lines = []
        for q, a in self.get(session_id):
            lines.append(f"用户：{q}")
            lines.append(f"客服：{a}")
        return "\n".join(lines)
