# 知选商城智能客服项目原理与实战指南

本文帮助你从“会运行”深入到“能修改、能排错、能扩展”。重点解释：

- 用户问题从页面到模型的完整调用链
- 文档如何被切片、向量化并建立索引
- 向量检索、BM25 和 RRF 如何协同
- RAG、Agent、订单工具和会话记忆的区别
- chain 模式多轮上下文的处理方式
- 如何配置、测试、调试、扩展和发布项目

---

## 1. 项目定位

这是一个电商客服 RAG Agent。它把本地业务文档、Embedding、混合检索、大语言模型和业务工具组合成一个多轮客服系统。

RAG 是 Retrieval-Augmented Generation，中文通常译为“检索增强生成”。系统不会把全部知识库直接发送给模型，而是先找到与问题最相关的文档片段，再让模型基于这些片段回答。

整体数据流：

~~~text
data/raw 文档
    |
    v
加载与 Markdown 切片
    |
    v
Embedding 向量化 -----> runtime/vectors.json
    |                         |
    |                         v
    +------------------> 混合检索
                         向量 + BM25 + RRF
                                  |
                                  v
                 知识库资料 + 会话历史 + 用户问题
                                  |
                                  v
                              LLM 生成
                                  |
                                  v
                         Gradio 或 CLI 展示
~~~

项目提供两种模式：

| 模式 | 工作方式 | 适合场景 |
| --- | --- | --- |
| agent | LLM 自主决定调用知识库工具、订单工具或直接回答 | 多工具客服 |
| chain | 固定执行查询改写、混合检索和 RAG 生成 | 可控、可解释的知识库问答 |

默认模式由 DEFAULT_MODE 决定，通常是 agent。

---

## 2. 目录结构和阅读顺序

~~~text
app.py                 Gradio Web 界面入口
cli.py                 命令行多轮问答入口
build_index.py         构建知识库索引
evaluate.py            检索命中率评测
requirements.txt       独立安装依赖
README.md              快速上手说明

data/raw/              原始 Markdown / TXT 知识库
data/orders_sample.json 模拟订单数据
runtime/vectors.json   运行时向量索引，不提交 Git

kb_core/config.py      配置和 .env 读取
kb_core/loader.py      文件加载和文件哈希
kb_core/splitter.py    标题感知和递归切片
kb_core/embeddings.py  Embedding provider
kb_core/store.py       JSON 向量库
kb_core/bm25.py        中文分词和 BM25
kb_core/retriever.py   混合检索和 RRF
kb_core/rag.py         RAG 检索与生成
kb_core/history.py     会话记忆
kb_core/agent.py       LangChain Agent
kb_core/order_tool.py  模拟订单工具
kb_core/llm.py         LLM 构造器
kb_core/service.py     统一门面服务
tests/                 离线单元测试
~~~

建议阅读顺序：

1. kb_core/config.py
2. kb_core/service.py
3. kb_core/loader.py 和 splitter.py
4. embeddings.py、store.py、bm25.py
5. retriever.py
6. rag.py
7. agent.py
8. app.py

service.py 是总调度入口，其他模块围绕它协作。

---

## 3. 启动时发生什么

运行：

~~~powershell
conda activate AI
python app.py
~~~

### 3.1 读取配置

config.py 使用 python-dotenv，只读取当前项目根目录的 .env，不会扫描或借用父目录中的配置文件。这样项目移动、克隆或部署到其他机器后，不依赖原来的目录结构。

配置优先级：

~~~text
启动 Python 进程前显式设置的系统环境变量
        |
项目根目录 .env 中的 LLM_* / EMBEDDING_* / RAG_*
        |
Settings 中的代码默认值
~~~

项目不再自动读取 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL 等旧字段。迁移已有 DeepSeek 配置时，应在项目自己的 .env 中使用：

~~~dotenv
LLM_API_KEY=你的本地密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
~~~

其中 LLM_API_KEY 没有可用的内置默认值，必须通过系统环境变量或项目 .env 提供。真实 .env 已被 .gitignore 排除，Git 中只保留不含密钥的 .env.example。

### 3.2 创建 AssistantService

app.py 顶层创建 svc = AssistantService()。

AssistantService 初始化时会：

1. 创建 Settings。
2. 创建 runtime 目录。
3. 创建 Embedding 客户端。
4. 创建 VectorStore，并尝试读取 runtime/vectors.json。
5. 创建 SessionMemory。
6. 如果向量库已有数据，就重建 BM25Index 和 HybridRetriever。

这里的“创建模型客户端”不等于已经调用远程模型。LLM、KnowledgeBase 和 Agent 都是懒加载的，通常第一次问答时才真正构造。

### 3.3 创建页面

build_ui() 创建三栏：

- 左栏：知识库状态、回答模式、索引操作、快捷问题。
- 中栏：聊天消息和输入框。
- 右栏：来源核验和响应状态。

发送消息后，Gradio 调用 app.py 的 chat()，再进入：

~~~python
svc.ask(message, session_id=session_id, mode=mode)
~~~

---

## 4. 知识库索引构建

首次运行或修改 data/raw 后执行：

~~~powershell
conda activate AI
python build_index.py
~~~

流程：

~~~text
data/raw/*.md / *.txt
        |
load_raw_documents()
        |
split_markdown()
        |
Document(page_content, metadata)
        |
embeddings.embed_documents()
        |
VectorStore.add_documents()
        |
runtime/vectors.json
        |
BM25Index + HybridRetriever
~~~

### 4.1 文档加载

loader.py 扫描 .md、.markdown 和 .txt 文件，读取 UTF-8 或 GB18030 编码。

每个切片包含 source、section、file_sha1 等元数据：

~~~python
{
    "source": "商品与规格.md",
    "section": "商品与规格 / X3 手机",
    "file_sha1": "..."
}
~~~

file_sha1 可以用于以后实现增量索引和文件变更检测。

### 4.2 Markdown 切片

splitter.py 使用两级策略：

1. 按 #、##、### 等标题识别章节路径。
2. 超长章节按中文标点递归切分，并保留 overlap。

例如：

~~~markdown
# 退换货政策
## 七天无理由
自签收起 7 天内可退。
~~~

会变成：

~~~text
【退换货政策 / 七天无理由】
自签收起 7 天内可退。
~~~

标题前缀非常重要。它使单独的文本块仍然知道自己属于哪个主题。

### 4.3 Embedding

Embedding 把文本转换为数值向量。语义相近的文本通常在向量空间中更接近。

项目支持：

| Provider | 用途 |
| --- | --- |
| openai / openai_compatible | Ollama、百炼或其他兼容接口 |
| huggingface | 本地 HuggingFace 模型 |
| debug | 无网络离线测试，效果较弱 |

构建索引和查询时必须尽量使用同一个 Embedding 模型，否则向量空间不一致，检索质量会明显下降。

### 4.4 JSON 向量库

项目没有依赖 Chroma 或 Milvus，而是把向量和文档保存为 JSON。每条记录大致如下：

~~~json
{
  "id": "chunk-0",
  "vector": [0.01, -0.02],
  "page_content": "【商品与规格 / X3】...",
  "metadata": {
    "source": "商品与规格.md",
    "section": "商品与规格 / X3",
    "chunk_id": "chunk-0",
    "file_sha1": "..."
  }
}
~~~

这种实现适合课程项目和小型本地知识库。数据规模变大后，应替换 VectorStore。

---

## 5. 混合检索原理

向量检索擅长语义近似，但对 X3、订单号和政策编号这类精确词不一定稳定。BM25 擅长关键词和型号匹配，但不理解同义表达。

因此 HybridRetriever 同时执行两路检索。

### 5.1 向量检索

VectorStore.search()：

1. 对查询做 Embedding。
2. 与全部保存向量计算余弦相似度。
3. 按相似度取 RAG_VECTOR_TOP_K 条。

### 5.2 BM25

BM25Index：

1. 优先用 jieba 分词。
2. 没有 jieba 时使用中文单字和二元组切分。
3. 根据词频、文档频率和文档长度计算分数。
4. 取 RAG_BM25_TOP_K 条。

### 5.3 RRF 融合

项目使用 Reciprocal Rank Fusion，而不是直接把两种分数相加：

~~~text
RRF(d) = 1 / (k + rank_vector(d))
       + 1 / (k + rank_bm25(d))
~~~

k 为 60。一个切片在两路结果中都靠前时，综合排名会更高。

两路结果通过稳定的 chunk_id 对齐和去重。

主要参数：

~~~dotenv
RAG_VECTOR_TOP_K=8
RAG_BM25_TOP_K=8
RAG_FINAL_TOP_K=5
~~~

调参建议：

- 型号或订单号召回差：提高 BM25 候选数。
- 同义问法召回差：检查真实 Embedding 是否可用。
- 最终上下文不足：提高 FINAL_TOP_K，但过大会引入噪音。

---

## 6. RAG 生成流程

KnowledgeBase.ask() 负责检索和最终生成：

~~~text
用户原始问题
    |
    +--> retrieval_query -> Embedding -> 混合检索 -> docs
    |
    +--> 原始问题 + 会话历史 + docs -> Prompt
                                             |
                                             v
                                            LLM
                                             |
                                             v
                              RagResult(answer, sources, docs)
~~~

### 6.1 检索问题和回答问题分开

当前实现明确区分：

- retrieval_query：用于召回资料，可以是改写后的问题。
- question：用户真正问出的原始问题，用于最终回答。

例如：

~~~text
第一轮：知选 X3 手机多少钱？
第二轮：它支持防水吗？
~~~

第二轮可以改写成“知选 X3 手机支持防水吗？”用于检索，但最终回答仍然面对用户原始问题“它支持防水吗？”。

### 6.2 会话历史

chain 模式会把历史放进最终 System Prompt：

~~~text
【当前会话历史】
用户：知选 X3 手机多少钱？
客服：X3 售价 4299 元。

【知识库资料】
...
~~~

所以这些问题可以使用会话历史回答：

- 我刚才问的第一条问题是什么？
- 你上一条回答是多少？
- 我们刚才说的商品是什么？

如果没有命中知识库但存在会话历史，RAG 仍会调用模型，而不是直接返回“没有检索到资料”。

### 6.3 模型失败兜底

LLM 调用失败时，KnowledgeBase 会返回检索资料摘要，避免 Web 请求整体崩溃。

常见原因：

- LLM_API_KEY 为空。
- LLM_BASE_URL 错误。
- 模型名称不支持。
- 网络超时。
- Embedding 服务不可用。

---

## 7. 两种问答模式

### 7.1 chain

入口：AssistantService.ask(..., mode="chain")。

流程：

~~~text
读取 session 历史
    |
查询改写 LLM 补全省略指代
    |
改写问题用于检索
    |
原始问题 + 历史 + 资料交给 RAG LLM
    |
保存本轮问答
~~~

优点是路径固定、容易调试、来源清晰。限制是不会自主调用订单工具。

### 7.2 agent

入口：EcommerceAgent.chat()。

Agent 会读取当前会话的历史消息，并自主选择：

1. knowledge_base_query：商品、政策、物流、发票、会员、售后。
2. check_order：从用户话术提取 XZ 开头的订单号，查询模拟订单。

流程：

~~~text
用户问题 + 历史
    |
LLM 判断意图
    |
    +--> 闲聊，直接回答
    +--> knowledge_base_query
    +--> check_order
    |
整理工具结果并生成客服回答
~~~

### 7.3 Agent 回退

如果 Agent 创建失败、模型不支持工具调用或网络异常，service.py 会尝试退回 chain 流程。回退后能继续进行知识库问答，但不具备订单工具能力。

---

## 8. 会话记忆

SessionMemory 使用：

~~~python
session_id -> deque[(question, answer)]
~~~

默认最多保留 12 轮。超过后最旧轮次会被移除。

Web 中的 session_id 来自 Gradio request.session_hash。如果前端历史为空，app.chat() 会清空后端对应会话，避免前后端状态不一致。

当前记忆的边界：

- 只存在当前 Python 进程内。
- 重启服务后丢失。
- 多进程之间不共享。
- 生产环境应替换为 Redis 或数据库。

点击“清空记忆”会删除当前会话，并重置页面右侧状态。

---

## 9. 配置

不要把真实密钥提交到 Git。典型 .env：

~~~dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=填入你的密钥
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.2

EMBEDDING_PROVIDER=debug
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=

RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=80
RAG_VECTOR_TOP_K=8
RAG_BM25_TOP_K=8
RAG_FINAL_TOP_K=5

DEFAULT_MODE=agent
~~~

### 9.1 Debug Embedding

没有 Embedding 服务时使用：

~~~dotenv
EMBEDDING_PROVIDER=debug
~~~

可以离线跑通索引和测试，但检索质量明显弱于真实语义模型。

### 9.2 Ollama

~~~dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=nomic-embed-text
~~~

百炼、硅基流动等兼容服务只需替换地址、密钥和模型名。

---

## 10. 从零安装和运行

~~~powershell
cd C:\Users\m1527\Desktop\AI\知识库Agent
conda activate AI
python -m pip install -r requirements.txt
~~~

网络较慢时：

~~~powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~

构建索引：

~~~powershell
python build_index.py
~~~

启动 Web：

~~~powershell
python app.py
~~~

默认端口由 app.py 中的 server_port 控制，当前默认 7860。端口被占用时，先查询并关闭占用 PID，或修改端口。

启动 CLI：

~~~powershell
python cli.py --mode agent
python cli.py --mode chain
python cli.py --mode chain --once "知选 X3 手机多少钱？"
~~~

---

## 11. 调试手册

### 11.1 知识库为空

~~~powershell
Get-ChildItem data\raw
Get-Item runtime\vectors.json
python build_index.py
~~~

如果切片数为 0，检查 data/raw 是否有 .md 或 .txt 文件。

### 11.2 检索效果差

按顺序检查：

1. 是否使用 debug Embedding。
2. 建库和查询是否使用相同 Embedding 模型。
3. Markdown 标题是否清晰。
4. 文档是否切得过大或过碎。
5. RAG_FINAL_TOP_K 是否太小。
6. evaluate.py 的命中率。

~~~powershell
python evaluate.py
~~~

### 11.3 模型失败

检查：

- LLM_API_KEY 是否为空。
- LLM_BASE_URL 是否正确。
- LLM_MODEL 是否被服务商支持。
- 网络是否可达。

不要把密钥粘贴到日志、Issue 或 GitHub。

### 11.4 Agent 失败但 chain 正常

通常说明：

- 模型不支持 tool calling。
- langchain 和 langgraph 版本不兼容。
- Agent 工具定义或模型配置错误。

先运行 chain 验证底层 RAG：

~~~powershell
python cli.py --mode chain
~~~

### 11.5 多轮不记忆

检查：

1. 是否使用同一个 session_id。
2. 是否点击了清空记忆。
3. 是否重启了 Python 进程。
4. 是否超过 max_session_turns。
5. 是否实际运行了最新代码。

### 11.6 端口被占用

~~~powershell
netstat -ano | Select-String ":7860"
Stop-Process -Id <PID> -Force
~~~

结束前确认 PID 属于本项目服务。

---

## 12. 测试

运行：

~~~powershell
conda activate AI
python -m pytest tests -q
~~~

当前测试覆盖：

- Debug Embedding。
- Markdown 标题切片和长文本切片。
- 中文 BM25。
- 向量库持久化。
- 混合检索和 RRF 去重。
- SessionMemory 截断和清空。
- RAG Prompt 是否包含会话历史。
- chain 是否正确传递原始问题、历史和检索改写。

单元测试尽量不依赖网络和真实 API Key，使用临时目录、Debug Embedding 和假模型。

---

## 13. 扩展方式

### 13.1 增加知识库

将 Markdown 或 TXT 放入 data/raw，然后重新运行 build_index.py。

文档建议：

- 使用清晰的 #、## 标题。
- 每个章节只讨论一个主题。
- 准确写出型号、政策编号和业务关键词。
- 避免把无关主题放在同一个巨大段落。

### 13.2 接入真实订单系统

修改 order_tool.py 的 check_order()，将本地 JSON 替换为 HTTP 或 RPC 请求。输入和输出接口可以保持不变。

生产环境还要增加：

- 用户身份校验。
- 订单查询权限。
- 超时和重试。
- 敏感信息脱敏。
- 审计日志。

### 13.3 替换向量库

保持 VectorStore 的主要接口，替换内部 JSON 实现。至少要支持：

- 添加文档和向量。
- 删除索引。
- 获取数量。
- 相似度搜索。
- 稳定 chunk_id。

### 13.4 增加 Rerank

当前是：

~~~text
向量召回 + BM25
        |
        v
RRF 合并
        |
        v
LLM
~~~

资料量变大后，可以加 Cross-Encoder 或 API Rerank：

~~~text
向量召回 + BM25
        |
        v
RRF 候选
        |
        v
Rerank 精排
        |
        v
LLM
~~~

### 13.5 持久化会话

将 SessionMemory 替换为 Redis 或数据库，实现多进程共享、服务重启恢复和会话过期。

---

## 14. Git 工作流

日常流程：

~~~powershell
cd C:\Users\m1527\Desktop\AI\知识库Agent
conda activate AI
git status
python -m pytest tests -q
git add .
git commit -m "描述本次修改"
git push
~~~

第一次连接 GitHub：

~~~powershell
git remote add origin https://github.com/<username>/<repository>.git
git branch -M main
git push -u origin main
~~~

注意：

- 终端输入原始 URL，不要输入 Markdown 的 [名称](URL)。
- .env 不应提交。
- runtime 是生成物，可以在目标机器重新构建。
- 提交前用 git status 检查密钥和日志。

---

## 15. 一次请求的代码追踪

以发送“X3 支持防水吗？”为例：

1. app.py 的 chat() 收到文本、历史和模式。
2. request.session_hash 变成会话 ID。
3. AssistantService.ask() 根据模式分流。
4. chain 模式读取 SessionMemory。
5. 查询改写补全省略指代。
6. KnowledgeBase 对检索问题做 Embedding。
7. HybridRetriever 执行向量检索和 BM25。
8. RRF 按 chunk_id 合并结果。
9. KnowledgeBase 组织资料、历史和原始问题。
10. ChatOpenAI 调用配置的模型。
11. 返回 answer、sources、docs 和状态字段。
12. app.py 将答案显示在聊天框，来源显示在右侧。
13. SessionMemory 保存本轮问题和答案。

定位问题时按层判断：

~~~text
页面无响应       -> app.py / Gradio
模式路由错误     -> service.py
没有召回资料     -> embeddings / store / bm25 / retriever
资料对但答案错   -> rag.py Prompt / LLM
Agent 不调工具   -> agent.py / tool calling
多轮不记忆       -> history.py / session_id / service.py
~~~

---

## 16. 架构边界和生产化路线

当前项目适合学习、原型和小规模内部工具，不是完整生产系统。

主要边界：

- JSON 向量库不适合大规模并发。
- 会话只保存在单进程内存。
- 没有认证和权限系统。
- 订单工具使用模拟数据。
- 缺少完整 Prompt 注入防护。
- 缺少成本、延迟和质量监控。
- RRF 后没有独立 Rerank。
- Web 启动方式是本地开发模式。

建议的生产化顺序：

1. 增加认证和权限隔离。
2. 将会话和向量索引迁移到持久化服务。
3. 增加日志、追踪、耗时和错误指标。
4. 建立 RAG 评测集和回归测试。
5. 增加 Prompt 注入、敏感信息和工具权限防护。
6. 再优化并发、缓存、流式输出和模型路由。

---

## 17. 最短掌握路径

如果想通过实验理解系统：

1. 修改 data/raw/商品与规格.md，重新构建索引，观察切片数。
2. 运行 evaluate.py，观察召回命中率。
3. 比较 chain 和 agent 的 CLI 输出。
4. 在 HybridRetriever.search() 中打印两路候选和 RRF 结果。
5. 在 rag.py 中打印最终 Prompt。
6. 修改 orders_sample.json，测试 Agent 的订单工具。
7. 将 Embedding 切换为 debug，观察离线能力边界。
8. 将 SessionMemory 换成持久化实现，理解内存状态和业务状态的区别。

掌握这条链路后，你就能独立替换模型、知识库、检索器、业务工具和前端界面。
