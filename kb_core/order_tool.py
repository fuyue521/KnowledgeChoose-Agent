"""
订单查询模拟工具：从 data/orders_sample.json 读取模拟订单数据。
用于演示 Agent 的多工具编排（知识库 + 业务系统查询）。
真实业务中可替换为调用订单中心的 HTTP/RPC 接口。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from kb_core.config import Settings


def load_orders(settings: Settings) -> List[Dict]:
    path: Path = settings.orders_file
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def extract_order_no(text: str) -> Optional[str]:
    """从用户话术中提取订单号（支持 XZ 开头 + 数字）。"""
    m = re.search(r"XZ\s?\d{6,}", text.upper())
    return m.group(0).replace(" ", "") if m else None


def _fmt_order(order: Dict) -> str:
    lines = [
        f"订单号：{order.get('order_no')}",
        f"状态：{order.get('status')}",
        f"商品：{order.get('product')}",
        f"金额：{order.get('amount')} 元",
        f"下单时间：{order.get('created_at')}",
    ]
    track = order.get("tracking")
    if track:
        lines.append(f"物流公司：{track.get('company')}")
        lines.append(f"物流单号：{track.get('no')}")
        lines.append(f"最新物流：{track.get('latest')}")
    return "\n".join(lines)


def to_tool_function(settings: Settings):
    """Agent 工具函数：入参为用户话术，返回订单信息文本。"""

    def check_order(text: str) -> str:
        orders = load_orders(settings)
        if not orders:
            return "订单系统暂不可用（未找到 data/orders_sample.json），请稍后再试。"
        order_no = extract_order_no(text)
        if not order_no:
            sample = "、".join(o["order_no"] for o in orders[:3])
            return f"没有识别到订单号，请提供 XZ 开头的订单号。可尝试示例订单：{sample}"
        for o in orders:
            if o["order_no"] == order_no:
                return _fmt_order(o)
        return f"未查询到订单 {order_no}，请核对订单号是否正确。"

    check_order.__name__ = "check_order"
    check_order.__doc__ = (
        "查询订单状态与物流信息。当用户询问“我的订单/物流到哪了/发货没有”等需要查订单的问题时使用。"
        "输入用户的整句话（会自动提取订单号）。"
    )
    return check_order
