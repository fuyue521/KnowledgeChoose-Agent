"""
知选商城 · 智能客服 RAG Agent —— Gradio Web 界面
启动：python app.py
"""
from __future__ import annotations

import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    raise SystemExit(
        "缺少 gradio，请先安装：conda activate AI && pip install \"gradio>=5.0,<6\""
    )

from kb_core.service import AssistantService

svc = AssistantService()

# 若知识库为空，引导先建索引
if svc.vector_store.count == 0:
    gr.Warning("知识库为空：请先关闭本界面，在终端运行 python build_index.py 构建索引后再启动。")

_MODE_CHOICES = [
    ("Agent 智能编排", "agent"),
    ("知识库问答", "chain"),
]


def _session_id(request: Optional[gr.Request]) -> str:
    return getattr(request, "session_hash", None) or "web-demo"


def _mode_value(mode_label: str) -> str:
    for label, value in _MODE_CHOICES:
        if label == mode_label:
            return value
    return "agent"


def _render_sources(result: dict) -> str:
    docs = result.get("docs") or []
    sources = result.get("sources") or []
    if docs:
        lines = ["### 参考来源"]
        for i, doc in enumerate(docs[:4], 1):
            source = doc.metadata.get("source", "未知来源")
            section = doc.metadata.get("section", "")
            score = doc.metadata.get("retrieval_score", doc.metadata.get("vector_score", ""))
            score_txt = f" · 相关度 {score}" if score != "" else ""
            title = source if not section else f"{source} / {section}"
            snippet = " ".join(doc.page_content.split())
            if len(snippet) > 150:
                snippet = snippet[:150].rstrip() + "..."
            lines.append(f"{i}. `{title}`{score_txt}")
            lines.append(f"   - {snippet}")
        return "\n".join(lines)
    if sources:
        lines = ["### 参考来源"]
        for source in sources:
            lines.append(f"- `{source}`")
        return "\n".join(lines)
    if result.get("fallback"):
        return "### 参考来源\n- 当前回答已自动回退到知识库兜底，没有额外工具来源。"
    if result.get("mode") == "agent":
        return "### 参考来源\n- Agent 回答已由工具内嵌整理，不单独展开来源。"
    return "### 参考来源\n- 当前回答未展示独立来源。"


def _render_meta(result: dict, session_id: str) -> str:
    mode = result.get("mode", "agent")
    route = result.get("route", mode)
    fallback = result.get("fallback", False)
    source_count = len(result.get("sources") or [])
    session_short = session_id[:8] if session_id else "web-demo"
    lines = [
        "### 本次响应",
        f"- 模式：`{mode}`",
        f"- 路由：`{route}`",
        f"- 会话：`{session_short}`",
        f"- 来源：`{source_count}` 个",
    ]
    if fallback:
        lines.append("- 状态：已自动回退到知识库兜底")
    return "\n".join(lines)


def chat(message: str, history: list, mode_label: str, request: Optional[gr.Request] = None):
    mode = _mode_value(mode_label)
    if not message or not message.strip():
        return (
            "请先告诉我您想咨询什么～",
            "### 参考来源\n- 等你发来问题，我会在这里列出来源。",
            "### 本次响应\n- 状态：等待输入",
        )

    session_id = _session_id(request)

    # UI 与后端记忆同步：若前端历史为空（新会话/已点清空），同步清空服务端记忆
    if not history:
        svc.clear_session(session_id)

    try:
        result = svc.ask(message, session_id=session_id, mode=mode)
        return result["answer"], _render_sources(result), _render_meta(result, session_id)
    except RuntimeError as e:
        logger.exception("问答失败")
        return (
            _friendly_error_message(e),
            "### 参考来源\n- 请求失败，未生成可引用来源。",
            "### 本次响应\n- 状态：请求失败\n- 建议：检查模型配置后重试",
        )
    except Exception as e:  # 其余未知异常也回给界面，便于调试
        logger.exception("问答失败")
        return (
            _friendly_error_message(e),
            "### 参考来源\n- 请求失败，未生成可引用来源。",
            "### 本次响应\n- 状态：请求失败\n- 建议：稍后重试或检查服务日志",
        )


def clear_current_session(request: Optional[gr.Request] = None):
    session_id = _session_id(request)
    svc.clear_session(session_id)
    return "### 参考来源\n- 当前会话已清空。", "### 本次响应\n- 状态：会话已重置"


def _friendly_error_message(error: Exception) -> str:
    name = type(error).__name__
    detail = str(error)
    lower_detail = detail.lower()
    if "connection" in lower_detail or "timeout" in lower_detail or name == "APIConnectionError":
        title = "模型服务暂时连不上"
        body = "这次请求没有成功生成回答。请检查网络、API 地址或密钥配置，然后再重试。"
    elif "auth" in lower_detail or "key" in lower_detail or "401" in lower_detail:
        title = "模型鉴权没有通过"
        body = "请检查 `.env` 中的 API Key 和模型服务配置。"
    else:
        title = "请求未完成"
        body = "服务处理时遇到异常，请稍后重试；如果持续出现，再查看终端日志定位原因。"
    return (
        '<div class="chat-error">'
        f"<strong>{title}</strong>"
        f"<p>{body}</p>"
        "</div>"
    )


def _kb_status() -> str:
    count = svc.vector_store.count
    if count == 0:
        return "知识库为空"
    return f"知识库就绪 · {count} 个切片"


def _status_html() -> str:
    state = "warning" if svc.vector_store.count == 0 else "success"
    return (
        f'<div class="status-line status-{state}">'
        '<span class="status-line-dot"></span>'
        f"<strong>{_kb_status()}</strong>"
        "</div>"
    )


def rebuild_index() -> str:
    try:
        count = svc.build_index(verbose=False)
    except Exception as e:
        logger.exception("索引重建失败")
        return (
            '<div class="status-line status-error">'
            '<span class="status-line-dot"></span>'
            f"<strong>索引重建失败</strong><span>{type(e).__name__}</span>"
            "</div>"
        )
    if count == 0:
        return (
            '<div class="status-line status-warning">'
            '<span class="status-line-dot"></span>'
            "<strong>未生成切片</strong><span>请检查 data/raw</span>"
            "</div>"
        )
    return (
        '<div class="status-line status-success">'
        '<span class="status-line-dot"></span>'
        f"<strong>索引已更新</strong><span>{count} 个切片</span>"
        "</div>"
    )


_PAGE_CSS = r"""
/* ===== 设计变量 ===== */
:root {
    --canvas: #f3f5f7;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --surface-raised: #ffffff;
    --ink: #1f2933;
    --ink-soft: #52606d;
    --muted: #667085;
    --hairline: #d8dee6;
    --hairline-soft: #e7ebf0;
    --accent: #c2410c;
    --accent-soft: #fff1eb;
    --info: #1d4ed8;
    --success: #147a52;
    --warning: #a15c00;
    --error: #b42318;
    --radius: 10px;
    --shadow-card: 0 1px 2px rgba(31, 41, 51, 0.04), 0 8px 20px rgba(31, 41, 51, 0.05);
}

html,
body {
    background: var(--canvas) !important;
}

/* ============================================================
   Gradio 5 的根容器类带版本后缀（如 gradio-container-5-50-0-dev0），
   且旧版样式通过 scoped 类注入；这里用「属性包含 + 后代」双保险，
   让布局跟随浏览器窗口宽度自适应（配合 Blocks fill_width=True）。
   ============================================================ */
.gradio-container,
[class*="gradio-container"] {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    background: transparent !important;
    padding: 18px clamp(16px, 3vw, 40px) 28px !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
        "PingFang SC", "Microsoft YaHei", sans-serif !important;
    color: var(--ink) !important;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    width: 100%;
    max-width: 1480px;
    margin: 0 auto;
    padding: 6px 2px 14px;
}

.app-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.app-brand-mark {
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: var(--ink);
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
}

.app-brand-name {
    font-size: 17px;
    font-weight: 650;
    line-height: 1.25;
    letter-spacing: 0;
    color: var(--ink);
}

.app-brand-sub {
    margin-top: 2px;
    font-size: 12px;
    line-height: 1.4;
    color: var(--muted);
}

.app-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    flex: 0 0 auto;
    padding: 7px 12px;
    border: 1px solid var(--hairline);
    border-radius: 8px;
    background: var(--surface);
    color: var(--ink-soft);
    font-size: 12px;
    font-weight: 500;
}

.app-header-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.header-context {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.4;
}

.app-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--success);
    box-shadow: 0 0 0 4px rgba(20, 122, 82, 0.12);
}

.workspace {
    width: 100% !important;
    max-width: 1480px;
    min-width: 0;
    margin: 0 auto !important;
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-sizing: border-box;
    box-shadow: var(--shadow-card);
}

.shell {
    display: grid !important;
    grid-template-columns: 248px minmax(0, 1fr) 280px;
    gap: 0 !important;
    align-items: stretch !important;
    min-width: 0 !important;
}

.sidebar {
    grid-column: 1;
    grid-row: 1;
    min-width: 0 !important;
    width: auto !important;
    padding: 0 12px 0 0 !important;
    border-right: 1px solid var(--hairline);
}

.sidebar > .side-section,
.context-pane > .context-head,
.context-pane > .context-section {
    flex: 0 0 auto !important;
}

.main-pane {
    grid-column: 2;
    grid-row: 1;
    min-width: 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    width: auto !important;
    padding: 0 16px !important;
}

.context-pane {
    grid-column: 3;
    grid-row: 1;
    min-width: 0 !important;
    width: auto !important;
    padding: 0 0 0 12px !important;
    border-left: 1px solid var(--hairline);
}

.main-toolbar {
    display: flex !important;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    min-width: 0;
    margin: 0 0 12px !important;
    padding: 2px 0 10px;
    border-bottom: 1px solid var(--hairline-soft);
}

.main-toolbar-copy {
    min-width: 0;
}

.main-toolbar-kicker,
.context-kicker {
    margin: 0 0 3px;
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.main-toolbar-title {
    margin: 0;
    color: var(--ink);
    font-size: 17px;
    font-weight: 650;
    line-height: 1.35;
}

.main-toolbar-note {
    flex: 0 0 auto;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.4;
    white-space: nowrap;
}

.context-head {
    padding: 2px 0 10px;
    border-bottom: 1px solid var(--hairline-soft);
}

.context-title {
    margin: 0;
    color: var(--ink);
    font-size: 15px;
    font-weight: 650;
    line-height: 1.4;
}

.context-section {
    margin: 12px 0 0;
    padding: 12px;
    border: 1px solid var(--hairline);
    border-radius: 10px;
    background: var(--surface);
}

.context-section .section-label {
    margin-bottom: 8px !important;
}

/* 聊天区必须收缩，消息内容在聊天框内部滚动，避免把输入框推到页面底部。 */
.chat-shell {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

.chat-shell > * {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    min-height: 0 !important;
}

.summary-row {
    align-items: stretch !important;
    gap: 12px !important;
    margin: 0 0 12px !important;
}

.summary-row > * {
    min-width: 0;
}

.summary-row .source-panel,
.summary-row .meta-panel {
    border: 1px solid var(--hairline);
    border-radius: 12px;
    background: var(--surface-raised);
    padding: 10px 12px !important;
    display: flex;
    flex-direction: column;
    min-height: 94px;
    max-height: 156px;
    overflow-y: auto;
    box-shadow: 0 1px 0 rgba(17, 17, 17, 0.02);
}

.context-section .source-panel,
.context-section .meta-panel {
    min-height: 0;
    max-height: 300px;
    overflow-y: auto;
}

.summary-row .source-panel {
    flex: 1 1 auto;
}

.summary-row .meta-panel {
    flex: 0 0 224px;
}

.side-section {
    width: 100%;
    padding: 10px !important;
    margin: 0 0 8px !important;
    border: 1px solid var(--hairline-soft);
    border-radius: 12px;
    background: var(--surface-soft);
}

.side-section:last-child {
    margin-bottom: 0 !important;
}

.side-section.compact {
    padding: 10px !important;
}

.side-intro {
    background: var(--surface-raised);
    border-color: var(--hairline);
}

.sidebar .side-section {
    border-radius: 8px;
}

.section-label {
    margin: 0 0 7px !important;
    font-size: 11px !important;
    font-weight: 650 !important;
    line-height: 1.4 !important;
    color: var(--muted) !important;
    letter-spacing: 0 !important;
    text-transform: uppercase;
}

.hero-copy {
    margin: 0 0 7px !important;
    font-size: 11px !important;
    line-height: 1.4 !important;
    color: var(--ink-soft) !important;
}

.status-line {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    margin: 0 !important;
    padding: 8px 10px;
    border: 1px solid var(--hairline);
    border-radius: 12px;
    background: var(--surface);
    color: var(--ink-soft);
    font-size: 11px;
    line-height: 1.35;
}

.status-line strong {
    color: var(--ink);
    font-weight: 650;
}

.status-line span:not(.status-line-dot) {
    color: var(--muted);
}

.status-line-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: var(--success);
    box-shadow: 0 0 0 4px rgba(17, 149, 93, 0.12);
    flex: 0 0 auto;
}

.status-warning .status-line-dot {
    background: var(--warning);
    box-shadow: 0 0 0 4px rgba(179, 106, 0, 0.14);
}

.status-error .status-line-dot {
    background: var(--error);
    box-shadow: 0 0 0 4px rgba(196, 28, 28, 0.12);
}

.stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.control-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: stretch;
}

.control-row > * {
    flex: 1 1 0;
    min-width: 0;
}

.quick-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    align-items: stretch;
}

.quick-grid > * {
    min-width: 0;
}

.quick-grid button {
    width: 100% !important;
    min-height: 42px !important;
    height: auto !important;
    padding: 7px 9px !important;
    border: 1px solid var(--hairline) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    border-radius: 10px !important;
    font-size: 11px !important;
    line-height: 1.25 !important;
    text-align: left !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    transition: background 0.14s ease, border-color 0.14s ease, transform 0.08s ease;
}

.quick-grid button:hover {
    background: var(--surface-soft) !important;
    border-color: #c7d1dc !important;
}

.quick-grid button:active,
.control-row button:active {
    transform: translateY(1px);
    border-color: var(--accent) !important;
}

.panel-note {
    margin: 2px 0 0 !important;
    font-size: 11px !important;
    line-height: 1.4 !important;
    color: var(--ink-soft) !important;
}

.control-row button {
    min-height: 38px !important;
    border: 1px solid var(--hairline) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
}

.control-row button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}

.main-pane .gradio-container {
    background: transparent !important;
}

.source-panel,
.meta-panel {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}

.source-panel h3,
.meta-panel h3 {
    margin: 0 0 8px !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
    color: var(--ink) !important;
    letter-spacing: 0 !important;
}

.source-panel p,
.source-panel li,
.meta-panel p,
.meta-panel li {
    margin: 0 !important;
    font-size: 12px !important;
    line-height: 1.55 !important;
    color: var(--ink-soft) !important;
    overflow-wrap: anywhere;
}

.source-panel ol,
.source-panel ul {
    margin: 0 !important;
    padding-left: 16px !important;
}

.source-panel li + li,
.meta-panel li + li {
    margin-top: 6px !important;
}

.source-panel code,
.meta-panel code {
    padding: 1px 5px;
    border-radius: 999px;
    background: #eef2f6;
    color: var(--ink);
    font-size: 11px;
}

.status-bar {
    align-items: center;
    gap: 12px;
    margin: 0 0 14px !important;
}

.kb-status {
    flex: 1 1 auto;
    min-width: 0;
}

.kb-status p {
    margin: 0;
    font-size: 12px;
    line-height: 1.45;
    color: var(--ink-soft);
}

.rebuild-btn {
    flex: 0 0 auto;
}

.mode-toolbar {
    align-items: center;
    gap: 12px;
    margin: 0 0 14px !important;
}

.mode-caption {
    flex: 0 0 auto;
    font-size: 12px;
    font-weight: 600;
    line-height: 1.4;
    color: var(--muted);
}

#mode-select {
    flex: 1 1 auto !important;
    width: 100%;
}

#mode-select .wrap {
    flex-direction: row !important;
    flex-wrap: wrap;
    gap: 8px !important;
}

#mode-select label {
    flex: 1 1 0 !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    margin: 0;
    padding: 6px 9px;
    border: 1px solid var(--hairline);
    border-radius: 10px;
    background: var(--surface);
    color: var(--ink-soft);
    font-size: 13px;
    font-weight: 500;
    line-height: 1.3;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}

#mode-select label.selected,
#mode-select label:has(input:checked) {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
}

#mode-select input[type="radio"] {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
}

#support-chatbot {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    min-height: 280px !important;
    height: min(560px, max(280px, calc(100vh - 260px))) !important;
    max-height: max(280px, calc(100vh - 260px)) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 16px !important;
    background: #fcfbf9 !important;
    overflow: hidden auto !important;
    padding: 16px 18px 12px !important;
    box-shadow: inset 0 1px 0 rgba(17, 17, 17, 0.02);
}

#support-chatbot .bubble-wrap,
#support-chatbot .message-wrap {
    width: 100%;
    min-width: 0;
    min-height: 0;
}

#support-chatbot .message-wrap {
    overflow-y: auto;
}

/* ===== flex-wrap 统一尺寸约束，避免短中文被压成逐字换行 ===== */
#support-chatbot .message-row .flex-wrap {
    max-width: min(84%, 760px) !important;
    min-width: 0 !important;
    width: auto !important;
    box-sizing: border-box;
}

#support-chatbot .message-row .flex-wrap > .message {
    font-size: 14px;
    line-height: 1.72;
    color: var(--ink);
    width: auto !important;
    max-width: 100% !important;
    min-width: 0;
    /* 删除 width:100%; 这一行 */
}

#support-chatbot .message-row {
    margin-bottom: 12px !important;
}

#support-chatbot .message-row .flex-wrap > .message.bot,
#support-chatbot .message-row .flex-wrap > .message.user {
    padding: 0;
    background: transparent !important;
    border: none !important;
}

#support-chatbot .message.bot .panel-full-width {
    display: block !important;
    width: max-content !important;
    max-width: 100% !important;
    padding: 11px 14px;
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 14px 14px 14px 5px !important;
    box-shadow: 0 1px 4px rgba(17, 17, 17, 0.04);
    color: var(--ink) !important;
    box-sizing: border-box;
    overflow-wrap: anywhere;
    word-break: normal;
}

#support-chatbot .message.user .panel-full-width {
    display: block !important;
    width: max-content !important;
    max-width: 100% !important;
    padding: 11px 14px;
    background: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 14px 14px 5px 14px !important;
    box-shadow: 0 1px 2px rgba(17, 17, 17, 0.08);
    color: #ffffff !important;
    box-sizing: border-box;
    overflow-wrap: anywhere;
    word-break: normal;
}

#support-chatbot .message.bot .panel-full-width .prose,
#support-chatbot .message.user .panel-full-width .prose {
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
}


#support-chatbot .message.user {
    margin-right: 8px !important;
}

#support-chatbot .message.bot {
    margin-left: 4px !important;
}

#support-chatbot .message.user,
#support-chatbot .message.user .prose,
#support-chatbot .message.user .prose * {
    color: #ffffff !important;
}

#support-chatbot .message .prose {
    font-size: 14px !important;
    line-height: 1.7 !important;
    min-width: 0;
    white-space: inherit !important;
    overflow-wrap: inherit !important;
    word-break: inherit !important;
}

#support-chatbot .message .prose p,
#support-chatbot .message .prose li {
    margin: 0 0 6px !important;
}

#support-chatbot .message .prose p:last-child,
#support-chatbot .message .prose li:last-child {
    margin-bottom: 0 !important;
}

#support-chatbot .message .prose code {
    padding: 1px 5px;
    border-radius: 999px;
    background: rgba(17, 17, 17, 0.06);
    color: inherit;
    font-size: 12px;
}

#support-chatbot .chat-error {
    border-left: 3px solid var(--error);
    padding-left: 10px;
}

#support-chatbot .chat-error strong {
    display: block;
    margin-bottom: 4px;
    color: var(--error) !important;
    font-size: 14px;
    line-height: 1.45;
}

#support-chatbot .chat-error p {
    margin: 0 !important;
    color: var(--ink-soft) !important;
    font-size: 13px;
    line-height: 1.65;
}

#support-chatbot .message.bot .panel-full-width:has(.chat-error) {
    border-color: rgba(196, 28, 28, 0.24) !important;
    background: #fff8f6 !important;
}

#support-chatbot .message-buttons button,
#support-chatbot .icon-button-wrapper button {
    min-width: 36px !important;
    min-height: 36px !important;
}

#support-input {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
    box-shadow: 0 1px 0 rgba(17, 17, 17, 0.02);
    overflow: hidden;
}

#support-input:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(194, 65, 12, 0.12);
}

#support-input textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 14px 14px 10px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: var(--ink) !important;
    resize: none;
    min-height: 56px;
}

#support-input textarea::placeholder {
    color: var(--muted) !important;
    opacity: 1;
}

.gradio-container button {
    border-radius: 14px !important;
    font-weight: 500 !important;
}

#support-input .submit-button {
    margin: 0 10px 10px 0 !important;
    min-height: 40px !important;
    min-width: 68px !important;
    padding: 0 18px !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important;
    background: var(--accent) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}

#support-input .submit-button:hover {
    background: #e94f00 !important;
    border-color: #e94f00 !important;
}

.gradio-container footer {
    display: none !important;
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 12px 8px 22px !important;
    }

    .app-header {
        padding-top: 4px;
    }

    .app-header-meta,
    .app-status {
        display: none;
    }

    .workspace {
        padding: 10px !important;
    }

    .main-pane {
        padding-left: 0 !important;
    }

    .main-toolbar-note {
        display: none;
    }

    .quick-grid {
        grid-template-columns: 1fr;
    }

    .quick-grid button {
        min-height: 48px !important;
    }

    #support-chatbot {
        min-height: 300px !important;
        height: min(520px, max(300px, calc(100vh - 300px))) !important;
        max-height: max(300px, calc(100vh - 300px)) !important;
    }

    #mode-select label {
        padding: 6px 8px;
        font-size: 12px;
    }
}

@media (max-width: 980px) {
    .workspace {
        min-width: 0;
    }

    .shell {
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: auto auto auto;
    }

    .sidebar {
        grid-column: 1;
        grid-row: 1;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        padding: 0 0 14px 0 !important;
        border-right: none;
        border-bottom: 1px solid var(--hairline);
    }

    .main-pane {
        grid-column: 1;
        grid-row: 2;
        padding: 14px 0 0 !important;
        min-width: 0 !important;
    }

    .context-pane {
        grid-column: 1;
        grid-row: 3;
        padding: 14px 0 0 !important;
        border-top: 1px solid var(--hairline);
        border-left: none;
    }

    .context-head {
        padding-top: 0;
    }

    .chat-shell {
        min-height: 0 !important;
    }
}

"""


_HEADER_HTML = """
<header class="app-header">
    <div class="app-brand">
        <span class="app-brand-mark">知</span>
        <div>
            <div class="app-brand-name">知选商城</div>
            <div class="app-brand-sub">客服运营台 · SUPPORT OPS</div>
        </div>
    </div>
    <div class="app-header-meta">
        <span class="header-context">知识库客服</span>
        <span class="app-status"><span class="app-status-dot"></span>系统在线</span>
    </div>
</header>
"""

_MAIN_TOOLBAR_HTML = """
<div class="main-toolbar">
    <div class="main-toolbar-copy">
        <div class="main-toolbar-kicker">Live support</div>
        <h1 class="main-toolbar-title">智能客服对话</h1>
    </div>
    <div class="main-toolbar-note">回答、路由和来源实时同步</div>
</div>
"""

_CONTEXT_HEAD_HTML = """
<div class="context-head">
    <div class="context-kicker">Context</div>
    <h2 class="context-title">运行上下文</h2>
</div>
"""

_CONFIRM_REBUILD_JS = """
() => {
    if (!confirm("重建索引会重新扫描 data/raw 并覆盖当前向量索引，确认继续？")) {
        throw new Error("cancelled");
    }
}
"""

_CONFIRM_CLEAR_JS = """
() => {
    if (!confirm("清空记忆会重置当前浏览器会话的对话历史，确认继续？")) {
        throw new Error("cancelled");
    }
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="知选商城 · 客服工作台", css=_PAGE_CSS, fill_width=True, fill_height=True) as demo:
        gr.HTML(_HEADER_HTML)
        with gr.Column(elem_classes=["workspace"]):
            with gr.Row(elem_classes=["shell"]):
                with gr.Column(scale=0, min_width=0, elem_classes=["sidebar"]):
                    with gr.Column(elem_classes=["side-section", "side-intro"]):
                        gr.HTML('<div class="section-label">工作台</div>')
                        gr.HTML('<div class="hero-copy">客服问答、来源核验和索引管理。</div>')
                        status = gr.HTML(_status_html())

                    with gr.Column(elem_classes=["side-section", "compact"]):
                        gr.HTML('<div class="section-label">回答模式</div>')
                        mode = gr.Radio(
                            choices=[m[0] for m in _MODE_CHOICES],
                            value=_MODE_CHOICES[0][0],
                            show_label=False,
                            container=False,
                            elem_id="mode-select",
                        )

                    with gr.Column(elem_classes=["side-section", "compact"]):
                        gr.HTML('<div class="section-label">操作</div>')
                        with gr.Row(elem_classes=["control-row"]):
                            rebuild = gr.Button("重建索引", variant="secondary", elem_classes=["rebuild-btn"])
                            clear_memory = gr.Button("清空记忆", variant="secondary")
                        gr.HTML('<div class="panel-note">需要重新读取资料或重置当前会话时使用。</div>')

                    with gr.Column(elem_classes=["side-section"]):
                        gr.HTML('<div class="section-label">常用问题</div>')
                        quick_prompts = [
                            "知选 X3 手机多少钱？",
                            "X3 支持防水吗？",
                            "耳机拆封了还能七天无理由退货吗？",
                            "满多少包邮？",
                            "能开发票吗？",
                            "帮我查订单 XZ202401150001",
                        ]
                        quick_buttons = []
                        with gr.Column(elem_classes=["quick-grid"]):
                            for prompt in quick_prompts:
                                btn = gr.Button(prompt, variant="secondary")
                                quick_buttons.append((btn, prompt))

                with gr.Column(scale=0, min_width=0, elem_classes=["context-pane"]):
                    gr.HTML(_CONTEXT_HEAD_HTML)
                    with gr.Column(elem_classes=["context-section"]):
                        gr.HTML('<div class="section-label">来源核验</div>')
                        source_panel = gr.Markdown(
                            "### 参考来源\n- 等待一次回答。",
                            elem_classes=["source-panel"],
                        )
                    with gr.Column(elem_classes=["context-section"]):
                        gr.HTML('<div class="section-label">响应状态</div>')
                        meta_panel = gr.Markdown(
                            "### 本次响应\n- 状态：等待输入。",
                            elem_classes=["meta-panel"],
                        )

                with gr.Column(scale=1, min_width=0, elem_classes=["main-pane"]):
                    gr.HTML(_MAIN_TOOLBAR_HTML)
                    chatbot = gr.Chatbot(
                        show_label=False,
                        container=False,
                        type="messages",
                        height=560,
                        elem_id="support-chatbot",
                        render=False,
                    )
                    textbox = gr.Textbox(
                        placeholder="请输入您的问题…",
                        show_label=False,
                        container=False,
                        lines=1,
                        max_lines=6,
                        autofocus=True,
                        submit_btn="发送",
                        elem_id="support-input",
                        render=False,
                    )
                    with gr.Column(elem_classes=["chat-shell"]):
                        gr.ChatInterface(
                            fn=chat,
                            chatbot=chatbot,
                            textbox=textbox,
                            additional_inputs=[mode],
                            additional_outputs=[source_panel, meta_panel],
                            type="messages",
                            title=None,
                            description=None,
                            fill_height=True,
                            fill_width=True,
                        )

                    for btn, prompt in quick_buttons:
                        btn.click(fn=lambda p=prompt: p, inputs=[], outputs=[textbox])
                    rebuild.click(fn=rebuild_index, inputs=[], outputs=[status], js=_CONFIRM_REBUILD_JS)
                    clear_memory.click(
                        fn=clear_current_session,
                        inputs=[],
                        outputs=[source_panel, meta_panel],
                        js=_CONFIRM_CLEAR_JS,
                    )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
