"""
文档加载与切片：
- loader：读取 data/raw 下的 .md/.txt 文件
- 切片：kb_core/splitter.py 自实现“标题感知 + 递归字符切片”（零外部依赖）
- 扩展：如需导入 PDF/Word 等格式，可在此文件新增对应 loader 分支
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from kb_core.config import Settings
from kb_core.splitter import split_markdown


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_text_file(path: Path) -> str:
    """读取 .md / .txt（自动处理 UTF-8 / GBK）。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def load_raw_documents(settings: Settings) -> List[Document]:
    """扫描 data/raw 目录，把每个文件切为带 source 元数据的文档块。"""
    raw_dir: Path = settings.raw_docs_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在：{raw_dir}，请先创建 data/raw 并放入资料")

    files = sorted(
        [p for p in raw_dir.iterdir() if p.suffix.lower() in (".md", ".txt", ".markdown")],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"知识库目录中没有 .md/.txt 文件：{raw_dir}")

    docs: List[Document] = []
    for path in files:
        text = load_text_file(path)
        file_docs = split_markdown(
            text,
            source=path.name,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        sha1 = _file_sha1(path)
        for d in file_docs:
            d.metadata["file_sha1"] = sha1
        docs.extend(file_docs)
    return docs
