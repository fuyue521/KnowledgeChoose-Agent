"""
一键构建知识库索引：
    python build_index.py
从 data/raw/ 读取 .md/.txt，切片 -> 向量化 -> 写入 runtime/vectors.json + 构建 BM25。
"""
from __future__ import annotations

import logging

from kb_core.service import AssistantService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    svc = AssistantService()
    count = svc.build_index()
    if count == 0:
        logger.warning("没有构建出任何切片，请检查 data/raw 目录内容。")
    else:
        logger.info("✅ 构建完成：共 %d 个切片已入库。现在可以运行 python app.py 启动问答界面。", count)


if __name__ == "__main__":
    main()
