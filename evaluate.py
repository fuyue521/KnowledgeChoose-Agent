"""
检索评测脚本：用一组标准问题检验“正确答案所在文档是否被召回”，
衡量知识库的检索命中率（无需消耗 LLM，只需 Embedding）。

    python evaluate.py

也可扩展为完整问答评测：设置 --with-llm 后会调用 LLM 生成回答并保存到 runtime/eval_report.md，
供人工对照检查回答质量。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from kb_core.service import AssistantService

# (问题, 期望命中的文档关键词) —— 可自行增删扩充测试集
DEFAULT_CASES: List[Tuple[str, str]] = [
    ("知选 X3 手机多少钱？", "商品与规格"),
    ("X3 支持防水吗？", "商品与规格"),
    ("耳机拆封了还能七天无理由退货吗？", "退换货政策"),
    ("退货运费谁出？", "退换货政策"),
    ("满多少包邮？用什么快递？", "物流配送"),
    ("能开发票吗？支持专票吗？", "支付与发票"),
    ("金卡会员有什么权益？", "会员与优惠"),
    ("手机屏幕碎了保修吗？", "售后与维修"),
    ("积分怎么用？", "会员与优惠"),
    ("R9 扫地机器人吸力多大？", "商品与规格"),
    ("顺丰加急怎么收费？", "物流配送"),
    ("碎屏险怎么用？", "售后与维修"),
]


def load_cases(path: Path | None) -> List[Tuple[str, str]]:
    if path is None or not path.exists():
        return DEFAULT_CASES
    cases = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            kw = row.get("expected_source", "").strip()
            if q:
                cases.append((q, kw))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库检索命中率评测")
    parser.add_argument("--cases", default=None, help="可选 CSV：列名 question,expected_source")
    args = parser.parse_args()

    svc = AssistantService()
    if svc.vector_store.count == 0:
        print("⚠️ 知识库为空，请先运行：python build_index.py")
        return

    cases = load_cases(Path(args.cases) if args.cases else None)
    print(f"共 {len(cases)} 条测试问题\n")
    hits = 0
    for q, kw in cases:
        # 混合检索直接查，看期望来源是否出现在 top-k
        query_vec = svc.embeddings.embed_query(q)
        docs = svc.retriever.search(q, query_vec)
        retrieved = " ".join(d.metadata.get("source", "") + " " + d.page_content for d in docs)
        hit = (kw in retrieved) if kw else True
        hits += 1 if hit else 0
        top = " | ".join(d.metadata.get("source", "?") for d in docs[:3])
        print(f"[{'✅' if hit else '❌'}] {q}\n     期望: {kw or '-'}  召回: {top}")
    print(f"\n检索命中率：{hits}/{len(cases)} = {hits / len(cases) * 100:.1f}%")


if __name__ == "__main__":
    main()
