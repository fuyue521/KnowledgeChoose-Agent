"""
单元测试：加载 / 切片 / 向量存储 / BM25 / 混合检索 / 会话记忆 / Embedding。
使用 debug Embedding，不需要网络与密钥。

运行方式（任选其一，均零外部依赖）：
    python -m unittest tests.test_core -v
    python -m pytest tests -v        （已安装 pytest 时）
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from kb_core.bm25 import BM25Index, tokenize
from kb_core.embeddings import DebugEmbeddings
from kb_core.history import SessionMemory
from kb_core.splitter import split_markdown
from kb_core.store import VectorStore, cosine_similarity

SAMPLE_MD = """# 退换货政策

## 七天无理由

自签收起 7 天内可退，运费买家承担。

## 运费说明

退货运费由买家承担。
"""


class TestEmbeddings(unittest.TestCase):
    def test_debug_embeddings_similar(self):
        emb = DebugEmbeddings(dim=128)
        a = emb.embed_query("退货运费谁出")
        b = emb.embed_documents(["退货需要买家承担运费", "今天的天气很好"])
        self.assertEqual(len(a), 128)
        self.assertGreater(cosine_similarity(a, b[0]), cosine_similarity(a, b[1]))


class TestSplitter(unittest.TestCase):
    def test_split_markdown_with_sections(self):
        docs = split_markdown(SAMPLE_MD, source="test.md", chunk_size=500, overlap=80)
        self.assertGreaterEqual(len(docs), 2)
        for d in docs:
            self.assertEqual(d.metadata["source"], "test.md")
        joined = "\n".join(d.page_content for d in docs)
        self.assertIn("七天无理由", joined)
        self.assertIn("运费", joined)
        # 标题前缀保留
        self.assertTrue(any("退换货政策 / 七天无理由" in d.page_content for d in docs))

    def test_long_text_recursive_split(self):
        long_text = ("# 标题\n\n" + ("这是一个很长的段落用于测试切片。") * 100)
        docs = split_markdown(long_text, source="long.md", chunk_size=200, overlap=30)
        self.assertGreater(len(docs), 1)
        for d in docs:
            self.assertLessEqual(len(d.page_content), 230)  # chunk + overlap 上限


class TestBM25(unittest.TestCase):
    def test_chinese_search(self):
        idx = BM25Index([
            "耳机拆封后不支持七天无理由退货",
            "手机支持防水和一百瓦快充",
            "扫地机器人支持自动回充",
        ])
        hits = idx.search("耳机退货", top_k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], 0)

    def test_exact_model_number_ranked(self):
        idx = BM25Index(["知选X3手机售价4299元", "耳机售价899元"])
        hits = idx.search("X3 多少钱", top_k=1)
        self.assertEqual(hits[0][0], 0)

    def test_tokenize(self):
        toks = tokenize("XZ20240115 订单 物流状态")
        self.assertGreater(len(toks), 0)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        import uuid

        from kb_core.config import PROJECT_ROOT, Settings

        tmp_root = PROJECT_ROOT / "runtime" / "ut_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tmp_root) / uuid.uuid4().hex
        self.tmp.mkdir(parents=True, exist_ok=True)

        self.cfg = Settings(
            runtime_dir=Path(self.tmp),
            vector_store_file=Path(self.tmp) / "vectors.json",
        )
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_persistence_and_search(self):
        vs = VectorStore(self.cfg)
        emb = DebugEmbeddings(dim=128)
        docs = [
            Document(page_content="支持七天无理由退货", metadata={"source": "a.md"}),
            Document(page_content="全场满99元包邮", metadata={"source": "b.md"}),
        ]
        vs.add_documents(docs, emb.embed_documents([d.page_content for d in docs]))
        self.assertEqual(vs.count, 2)

        # 重新加载（模拟重启）
        vs2 = VectorStore(self.cfg)
        self.assertEqual(vs2.count, 2)
        top = vs2.search(emb.embed_query("满多少包邮"), top_k=1)
        self.assertIn("包邮", top[0].page_content)

        vs2.delete_all()
        self.assertEqual(VectorStore(self.cfg).count, 0)


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        import uuid

        from kb_core.config import PROJECT_ROOT, Settings

        tmp_root = PROJECT_ROOT / "runtime" / "ut_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tmp_root) / uuid.uuid4().hex
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.cfg = Settings(
            runtime_dir=Path(self.tmp),
            vector_store_file=Path(self.tmp) / "vectors.json",
            vector_top_k=5,
            bm25_top_k=5,
            final_top_k=3,
        )
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self):
        from kb_core.bm25 import BM25Index
        from kb_core.retriever import HybridRetriever

        vs = VectorStore(self.cfg)
        emb = DebugEmbeddings(dim=128)
        docs = [
            Document(page_content="知选X3手机支持IP68防水，售价4299元", metadata={"source": "商品.md"}),
            Document(page_content="知选X3手机支持一百瓦快充", metadata={"source": "商品.md"}),
            Document(page_content="耳机拆封后不支持七天无理由退货", metadata={"source": "退货.md"}),
        ]
        vs.add_documents(docs, emb.embed_documents([d.page_content for d in docs]))
        bm25 = BM25Index([d.page_content for d in vs.all_documents()])
        retriever = HybridRetriever(self.cfg, vs, bm25)
        retriever.refresh()
        return vs, emb, retriever

    def test_fusion_returns_dedup_chunk_ids(self):
        vs, emb, retriever = self._build()
        query = "X3 手机 防水 多少钱"
        qvec = emb.embed_query(query)
        results = retriever.search(query, qvec)
        self.assertGreaterEqual(len(results), 1)
        ids = [r.metadata.get("chunk_id") for r in results]
        self.assertTrue(all(ids), "检索结果应带 chunk_id")
        self.assertEqual(len(ids), len(set(ids)), "chunk_id 不应重复")
        self.assertTrue(all("retrieval_score" in r.metadata for r in results))

    def test_refresh_after_reload(self):
        vs, emb, retriever = self._build()
        # 模拟重启后 VectorStore 重建、retriever 重新 refresh
        vs2 = VectorStore(self.cfg)
        from kb_core.bm25 import BM25Index
        from kb_core.retriever import HybridRetriever

        bm25 = BM25Index([d.page_content for d in vs2.all_documents()])
        retriever2 = HybridRetriever(self.cfg, vs2, bm25)
        retriever2.refresh()
        results = retriever2.search("耳机退货", emb.embed_query("耳机退货"))
        # “退货”关键词命中退货文档，应排在最前
        self.assertEqual(results[0].metadata["source"], "退货.md")


class TestSessionMemory(unittest.TestCase):
    def test_trim(self):
        mem = SessionMemory(max_turns=3)
        for i in range(5):
            mem.add("s1", f"q{i}", f"a{i}")
        history = mem.get("s1")
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0][0], "q2")  # 最旧的被挤出

    def test_clear(self):
        mem = SessionMemory()
        mem.add("s1", "q", "a")
        mem.clear("s1")
        self.assertEqual(mem.get("s1"), [])


class TestConversationContext(unittest.TestCase):
    def test_rag_prompt_includes_history_when_retrieval_misses(self):
        import uuid

        from kb_core.config import PROJECT_ROOT, Settings
        from kb_core.rag import KnowledgeBase

        tmp_root = PROJECT_ROOT / "runtime" / "ut_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tmp_root) / uuid.uuid4().hex
        tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))

        cfg = Settings(
            runtime_dir=tmp,
            vector_store_file=tmp / "vectors.json",
        )
        vs = VectorStore(cfg)
        vs.add_documents(
            [Document(page_content="商品资料", metadata={"source": "商品.md"})],
            [[1.0, 0.0]],
        )

        class FakeEmbeddings:
            def embed_query(self, _query):
                return [1.0, 0.0]

        class EmptyRetriever:
            def search(self, *_args, **_kwargs):
                return []

        class FakeLLM:
            def __init__(self):
                self.messages = None

            def invoke(self, messages):
                self.messages = messages
                return SimpleNamespace(content="你第一条问题是：知选 X3 手机多少钱？")

        llm = FakeLLM()
        kb = KnowledgeBase(cfg, vs, EmptyRetriever(), FakeEmbeddings())
        kb._llm = llm
        result = kb.ask(
            "我所说的第一条问题是什么？",
            history=[("知选 X3 手机多少钱？", "X3 售价 4299 元。")],
            retrieval_query="当前对话第一条问题",
        )

        self.assertIn("知选 X3 手机多少钱？", result.answer)
        system_prompt = llm.messages[0].content
        self.assertIn("当前会话历史", system_prompt)
        self.assertIn("用户：知选 X3 手机多少钱？", system_prompt)
        self.assertEqual(llm.messages[-1].content, "我所说的第一条问题是什么？")

    def test_chain_mode_forwards_history_to_knowledge_base(self):
        from kb_core.rag import RagResult
        from kb_core.service import AssistantService

        class FakeKnowledgeBase:
            def __init__(self):
                self.calls = []

            def ask(self, question, **kwargs):
                self.calls.append((question, kwargs))
                return RagResult(answer="基于会话历史的回答")

        service = AssistantService.__new__(AssistantService)
        service.settings = SimpleNamespace(default_mode="chain")
        service.memory = SessionMemory()
        service.memory.add("s1", "第一条问题", "第一条回答")
        service._kb = FakeKnowledgeBase()
        service._rewrite_question = lambda _question, _session_id: "检索改写问题"

        result = service.ask("我所说的第一条问题是什么？", session_id="s1", mode="chain")

        question, kwargs = service._kb.calls[0]
        self.assertEqual(question, "我所说的第一条问题是什么？")
        self.assertEqual(kwargs["retrieval_query"], "检索改写问题")
        self.assertEqual(kwargs["history"], [("第一条问题", "第一条回答")])
        self.assertEqual(result["answer"], "基于会话历史的回答")


if __name__ == "__main__":
    unittest.main()
