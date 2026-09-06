"""
自实现的中文 Markdown 切片器（零外部依赖）。

设计动机：`langchain-text-splitters` 属可选依赖，为避免网络/环境问题，
这里自实现两段式切片，效果等价于官方两件套：
  1) 标题感知切片：按 # / ## / ### 切出章节块，保留“标题路径”作为上下文前缀；
  2) 递归字符切片：超长章节再按中文标点迭代切分，带 overlap。
换用官方实现时，替换 split_markdown() 内部即可。
"""
from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.documents import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# 中文友好的分隔符（优先级从高到低）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def _iterative_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """迭代式字符切片：按分隔符把超长文本切碎，块间尽量保留 overlap。"""
    chunks: List[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        cut = -1
        for sep in _SEPARATORS:
            idx = remaining.rfind(sep, 0, chunk_size)
            if idx > 0:
                cut = idx + len(sep)
                break
        if cut <= 0:  # 找不到分隔符，硬切
            cut = chunk_size
        head = remaining[:cut]
        if head.strip():
            chunks.append(head)
        # overlap 不能越过文本起点，否则无进展
        start = cut - overlap if cut - overlap > 0 else cut
        remaining = remaining[start:]
    if remaining.strip():
        chunks.append(remaining)
    return chunks


def split_markdown(text: str, source: str, chunk_size: int = 500, overlap: int = 80) -> List[Document]:
    """Markdown 文档 -> 标题感知切片列表（每个 Document 带 source 元数据）。"""
    lines = text.splitlines()
    # 按标题行切块：blocks = [(标题路径, 正文)]
    blocks: List[Tuple[str, str]] = []
    cur_path: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        body = "\n".join(buf).strip()
        if body:
            prefix = " / ".join(cur_path) if cur_path else ""
            blocks.append((prefix, body))
        buf = []

    for raw in lines:
        line = raw.rstrip()
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # 标题层级只进不退：同级覆盖，更高级截断
            cur_path = cur_path[: level - 1]
            cur_path.append(title)
            continue
        buf.append(line)
    flush()

    docs: List[Document] = []
    for prefix, body in blocks:
        content = f"【{prefix}】\n{body}" if prefix else body
        if len(content) <= chunk_size:
            docs.append(Document(page_content=content, metadata={"source": source, "section": prefix}))
            continue
        for sub in _iterative_split(content, chunk_size, overlap):
            docs.append(Document(page_content=sub, metadata={"source": source, "section": prefix}))
    return docs
