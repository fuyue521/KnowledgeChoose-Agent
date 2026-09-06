# 交接文档（HANDOFF）

> 用途：电商客服 RAG Agent（LangChain 1.x）作业项目——当前进度、剩余待办与执行命令。
> 代码已全部完成并通过验证；依赖、配置、启动收尾已于 2026-09-06 完成。

## 一、项目信息

| 项 | 值 |
|---|---|
| 项目根目录 | `C:\Users\m1527\Desktop\AI\知识库Agent` |
| conda 环境 | `AI`（python 位于 `C:\Users\m1527\.conda\envs\AI\python.exe`） |
| LangChain 版本 | 1.3.18（**1.x 新 API**：`create_agent`；旧 `AgentExecutor` 已移除，勿照抄 0.3 教程） |
| 外层 .env | `C:\Users\m1527\Desktop\AI\.env`（已含 `DEEPSEEK_API_KEY`，项目 config 会自动复用，无需重填） |

## 二、已完成（已验证 ✅）

- 全部代码：`kb_core/`（15 模块）+ `app.py` / `cli.py` / `build_index.py` / `evaluate.py` / `tests/test_core.py`
- 知识数据：`data/raw/` 6 份客服 Markdown + `data/orders_sample.json`
- 离线验证结果：
  - `python -m pytest tests -q` → **11/11 通过**
  - `python build_index.py` → 25 个切片入库（现为百炼 1024 维）
  - `python evaluate.py`（百炼 embedding）→ 检索命中率 **12/12 = 100%**
  - 真实端到端：chain 问答（带来源）、Agent 订单查询、多轮对话记忆 均通过
- 依赖已装：`jieba 0.42.1` 已成功装入 AI 环境
- 外层 `.env` 复用验证：`LLM_API_KEY` 自动取自 `DEEPSEEK_API_KEY`，base_url 自动补 `/v1`
- 收尾验证（2026-09-06）：`gradio 5.50.0` / `pytest 9.1.1` 已装；项目 `.env` 已切换为百炼
  `qwen3.7-text-embedding`（1024 维）；chain 与 Agent 真实问答均通过；Gradio 页面已在
  `http://127.0.0.1:7860` 运行
- 代码体检（2026-09-06，11/11 测试通过）：
  - `config.py`：外层 `.env` 只“借读” DEEPSEEK_*，不再整包加载 —— 避免 LANGSMITH_TRACING
    等外层开关污染本项目进程（此前会导致每次 LLM 调用后向 LangSmith 反复重试上传、拖慢响应）
  - `store.py` / `retriever.py`：向量相似度改为完整余弦（不假定已归一化）；每个切片带稳定
    `chunk_id`，双路召回按 id 对齐去重（修复内容重复切片错配）
  - `rag.py`：复用同一 Embedding 实例（此前每次提问都会重建模型）
  - `service.py` / `agent.py`：Agent 失败回退逻辑细化、共享 LLM 实例
  - `app.py`：UI 清空/新会话时同步清空服务端记忆
  - 新增混合检索集成测试；全套 11/11 通过
  - 已用百炼 embedding 重建 `runtime/vectors.json`（1024 维）

## 三、依赖安装（本次已装好；重建环境时重跑以下命令）

```bash
conda activate AI
# 网络走阿里云 HTTP 镜像（HTTPS 在本机不通，必须用 http + --trusted-host）
pip install "gradio>=5.0,<6" pytest -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

> 说明：切片器 / BM25 / 向量库均为项目自实现，**不需要** langchain-text-splitters / rank-bm25 / chromadb。
> gradio 用于 Web 界面；pytest 可选（不装也能用 unittest 跑测试）。
> **版本提醒**：不要装 gradio 4.x。它的 gradio-client 把 websockets 锁在 `<13`，会与
> langgraph-sdk/langsmith 需要的 `websockets>=14/15` 冲突，导致 LangGraph 导入报错；
> 当前环境已按 5.x 安装验证（gradio 5.50.0 + websockets 15.0.1）。
> 注意本机 pip 需在能写 site-packages 的权限下执行（沙箱内需全权限，正常终端无此问题）。

## 四、环境配置（.env）

项目根目录下建 `.env`（模板：`.env.example`），**LLM 可不填**（自动读外层 `C:\Users\m1527\Desktop\AI\.env` 的 DEEPSEEK_*），需要配置的是 **Embedding**：

当前项目 `.env` 已配置百炼业务空间的 OpenAI 兼容地址与 `qwen3.7-text-embedding` 模型；
若之后改用其他 Embedding，需要重跑 `python build_index.py` 重建索引。

DeepSeek 不提供 embedding 接口，三选一：

- 方案 A（推荐，离线免费）：本机 Ollama
  ```bash
  # 另装 Ollama 后：
  ollama pull nomic-embed-text
  ```
  `.env`：
  ```
  EMBEDDING_PROVIDER=openai
  EMBEDDING_BASE_URL=http://localhost:11434/v1
  EMBEDDING_API_KEY=ollama
  EMBEDDING_MODEL=nomic-embed-text
  ```
- 方案 B：阿里云百炼
  ```
  EMBEDDING_PROVIDER=openai
  EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  EMBEDDING_API_KEY=sk-百炼密钥
  EMBEDDING_MODEL=text-embedding-v3
  ```
- 方案 C：不配 embedding → 代码自动降级 debug 模式（离线可跑通全流程，检索效果一般，仅演示）

## 五、启动步骤（当前 Web 已在运行；重启或新环境按下面执行）

```bash
conda activate AI
cd C:\Users\m1527\Desktop\AI\知识库Agent

# 1) 构建索引（换知识文档后需重跑）
python build_index.py

# 2a) Web 界面（需先装 gradio）
python app.py              # 打开 http://127.0.0.1:7860

# 2b) 或命令行验证（jieba 已装好，现在就能跑 chain 模式）
python cli.py --mode agent
python cli.py --mode chain
```

## 六、可以试的问题

- 知选 X3 手机多少钱？支持防水吗？
- 耳机拆封了还能七天无理由退货吗？
- 帮我查订单 XZ202401150001 的物流（测 Agent 订单工具）
- 金卡会员有什么权益？（多轮：再问"积分怎么用"测记忆）
- 你好，在吗？（测闲聊不调工具）

## 七、遗留小项（可选清理）

- `runtime/` 下有沙箱残留的空目录 `ut_probe/`、`ut_tmp/`（ACL 锁定删不掉，已被 .gitignore 覆盖，不影响运行；正常终端可手动 `rmdir /s /q` 清理）
- `runtime/vectors.json` 是已构建的索引产物，删除后重跑 `build_index.py` 即可

## 八、如需扩展（作业加分方向）

1. 换 Chroma：替换 `kb_core/store.py` 的 `VectorStore` 内部实现（接口已预留）
2. 接真实订单 API：改 `kb_core/order_tool.py` 的 `check_order`
3. 支持 PDF：装 `pypdf`，在 `kb_core/loader.py` 加 PDF 分支
4. RAGAS 自动化评测 / 加 Rerank / 换 LangGraph 自定义状态机
