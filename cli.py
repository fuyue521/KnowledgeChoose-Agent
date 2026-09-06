"""
命令行问答入口（无需界面，便于快速验证）：
    python cli.py --mode agent
    python cli.py --mode chain
输入 exit / quit 退出。
"""
from __future__ import annotations

import argparse
import sys

from kb_core.service import AssistantService


def main() -> None:
    parser = argparse.ArgumentParser(description="知选商城 RAG Agent 命令行问答")
    parser.add_argument("--mode", default="agent", choices=["agent", "chain"], help="问答模式")
    parser.add_argument("--once", default=None, help="单次提问后退出（供脚本调用）")
    args = parser.parse_args()

    svc = AssistantService()
    if svc.vector_store.count == 0:
        print("⚠️ 知识库为空，请先运行：python build_index.py")
        sys.exit(1)

    session_id = "cli"
    if args.once:
        try:
            result = svc.ask(args.once, session_id=session_id, mode=args.mode)
        except RuntimeError as e:
            print(f"⚠️ {e}")
            print("   → 请检查 .env 配置（复制 .env.example 为 .env 并填写 LLM_API_KEY 等）。")
            return
        print("\n" + result["answer"] + "\n")
        return

    print("=" * 60)
    print("知选商城智能客服 | 模式：", args.mode)
    print("输入问题开始，输入 exit 退出。示例：X3 手机多少钱？")
    print("=" * 60)
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "退出"):
            break
        if not q:
            continue
        try:
            result = svc.ask(q, session_id=session_id, mode=args.mode)
        except RuntimeError as e:
            print(f"\n⚠️ {e}")
            print("   → 请检查 .env 配置（复制 .env.example 为 .env 并填写 LLM_API_KEY 等）。")
            continue
        print(f"\n客服：{result['answer']}")


if __name__ == "__main__":
    main()
