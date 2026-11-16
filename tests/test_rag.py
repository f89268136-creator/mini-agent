"""RAG v2 离线测试：切块、持久化、混合检索、索引迁移与工具。

全部使用 HashingEmbedder，不访问网络、不需要 API 密钥。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from mini_agent.agent.tools.knowledge import (
    KnowledgeBase,
    _build_chunks,
    _mmr_select,
    configure_knowledge_base,
    knowledge_delete,
    knowledge_ingest,
    knowledge_ingest_dir,
    knowledge_list,
    knowledge_search,
    knowledge_stats,
    reset_knowledge_base,
)
from mini_agent.agent.tools.policy import configure_policy, workspace_root
from mini_agent.config import ConfigError, MiniConfig
from mini_agent.providers.embeddings import Embedder, HashingEmbedder, build_embedder

_DOC = (
    "# 上线流程\n\n"
    "## 准备\n\n"
    "先把代码合并到 main，然后运行完整测试。测试通过后构建镜像。\n\n"
    "## 部署\n\n"
    "使用 blue-green 方式部署，先切百分之十流量观察错误率。\n\n"
    "## 回滚\n\n"
    "如果错误率超过百分之一，立即回滚到上一个版本。\n"
)


class _BrokenEmbedder(Embedder):
    model = "broken"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")


class RagConfigTests(unittest.TestCase):
    def test_defaults_and_backend_factory(self) -> None:
        config = MiniConfig()
        self.assertEqual(config.knowledge_dir, ".mini_agent_knowledge")
        self.assertEqual(config.embedding_backend, "openai")
        self.assertEqual(config.knowledge_top_k, 4)
        self.assertIsInstance(
            build_embedder(MiniConfig(embedding_backend="hashing")), HashingEmbedder
        )

    def test_invalid_knowledge_settings_are_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            MiniConfig(embedding_backend="unknown")
        with self.assertRaises(ConfigError):
            MiniConfig(knowledge_chunk_size=100, knowledge_chunk_overlap=100)
        with self.assertRaises(ConfigError):
            MiniConfig(knowledge_top_k=21)
        with self.assertRaises(ConfigError):
            MiniConfig(knowledge_min_score=1.5)
        with self.assertRaises(ConfigError):
            MiniConfig(knowledge_lexical_weight=-0.1)
        with self.assertRaises(ConfigError):
            MiniConfig(knowledge_mmr_lambda=2.0)
        with self.assertRaises(ConfigError):
            MiniConfig(knowledge_dir="  ")


class ChunkingTests(unittest.TestCase):
    @staticmethod
    def _chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
        return [piece for piece, *_ in _build_chunks(text, chunk_size, overlap)]

    def test_paragraphs_are_preserved_and_limited(self) -> None:
        chunks = self._chunk(_DOC, 80, 15)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        joined = "\n".join(chunks)
        for phrase in ("完整测试", "blue-green", "回滚到上一个版本"):
            self.assertIn(phrase, joined)

    def test_overlap_repeats_tail_of_previous_chunk(self) -> None:
        text = "第一句内容比较长。" * 20
        chunks = self._chunk(text, 40, 10)
        self.assertGreater(len(chunks), 1)
        tail = chunks[0][-8:]
        self.assertIn(tail, chunks[1])

    def test_long_text_all_chunks_within_budget(self) -> None:
        chunks = self._chunk("这是一个很长的段落。" * 100, 100, 20)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_empty_text(self) -> None:
        self.assertEqual(self._chunk("   \n\n  ", 100, 10), [])


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index_dir = Path(self.temp.name)

    def _kb(self, dims: int = 64) -> KnowledgeBase:
        return KnowledgeBase(
            HashingEmbedder(dims),
            chunk_size=80,
            overlap=15,
            index_dir=self.index_dir,
        )

    def test_index_survives_restart_and_uses_compact_vectors(self) -> None:
        kb = self._kb()
        count = asyncio.run(kb.ingest(_DOC, source="ops.md", replace_source=True))
        self.assertGreater(count, 0)
        self.assertGreater(asyncio.run(kb.ingest("附加备注：周五冻结发布。", source="notes.md")), 0)

        index_file = self.index_dir / "index.jsonl"
        self.assertTrue(index_file.exists())
        lines = index_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(kb) + 1)
        meta = json.loads(lines[0])
        self.assertEqual(meta["type"], "meta")
        self.assertEqual(meta["version"], 2)
        self.assertEqual(meta["dims"], 64)
        record = json.loads(lines[1])
        self.assertNotIsInstance(record.get("vec"), list)
        self.assertTrue(record.get("vec"))
        self.assertGreaterEqual(record.get("start_line", 0), 1)
        self.assertGreaterEqual(record.get("end_line", 0), record["start_line"])

        reloaded = self._kb()
        self.assertEqual(len(reloaded), len(kb))
        self.assertEqual(reloaded.stats()["dims"], 64)
        self.assertFalse(reloaded.stats()["needs_reindex"])
        hits = asyncio.run(reloaded.search_hits("blue-green 部署", top_k=2))
        self.assertTrue(hits)
        self.assertTrue(any("blue-green" in hit.text for hit in hits))

        self.assertEqual(list(self.index_dir.glob("*.tmp")), [])

    def test_delete_source_only_removes_that_source(self) -> None:
        kb = self._kb()
        asyncio.run(kb.ingest(_DOC, source="ops.md"))
        asyncio.run(kb.ingest("备注：周五冻结发布。", source="notes.md"))
        removed = asyncio.run(kb.delete_source("ops.md"))
        self.assertGreater(removed, 0)
        self.assertEqual({item["source"] for item in kb.list_sources()}, {"notes.md"})

        reloaded = self._kb()
        self.assertEqual({item["source"] for item in reloaded.list_sources()}, {"notes.md"})

    def test_corrupt_line_is_skipped(self) -> None:
        kb = self._kb()
        asyncio.run(kb.ingest(_DOC, source="ops.md"))
        index_file = self.index_dir / "index.jsonl"
        with index_file.open("a", encoding="utf-8") as handle:
            handle.write("{not-json}\n")
        reloaded = self._kb()
        self.assertGreater(len(reloaded), 0)
        self.assertEqual(reloaded.stats()["sources"], 1)

    def test_clear_removes_all_chunks(self) -> None:
        kb = self._kb()
        asyncio.run(kb.ingest(_DOC, source="ops.md"))
        removed = asyncio.run(kb.clear())
        self.assertGreater(removed, 0)
        self.assertEqual(len(kb), 0)
        self.assertEqual(len(self._kb()), 0)


class RetrievalTests(unittest.TestCase):
    def _kb(self, **kwargs) -> KnowledgeBase:
        options = {
            "chunk_size": 80,
            "overlap": 10,
            "lexical_weight": 0.5,
            "mmr_lambda": 0.7,
        }
        options.update(kwargs)
        return KnowledgeBase(HashingEmbedder(128), **options)

    def test_lexical_only_retrieval_without_vectors(self) -> None:
        kb = self._kb(lexical_weight=1.0)
        asyncio.run(kb.ingest(_DOC, source="ops.md"))
        hits = asyncio.run(kb.search_hits("回滚上一个版本", top_k=2))
        self.assertTrue(hits)
        self.assertIn("回滚", hits[0].text)
        self.assertFalse(kb.last_search_degraded)

    def test_min_score_filters_unrelated_query(self) -> None:
        kb = self._kb(lexical_weight=1.0, min_score=0.2)
        asyncio.run(kb.ingest("星火项目每周三晚上部署。", source="a.md"))
        self.assertEqual(asyncio.run(kb.search_hits("量子纠缠拓扑绝缘体", top_k=3)), [])

    def test_embedding_failure_falls_back_to_lexical(self) -> None:
        kb = self._kb(lexical_weight=0.4)
        asyncio.run(kb.ingest(_DOC, source="ops.md"))
        kb.embedder = _BrokenEmbedder()
        hits = asyncio.run(kb.search_hits("回滚版本", top_k=2))
        self.assertTrue(hits)
        self.assertTrue(kb.last_search_degraded)

    def test_mmr_prefers_diverse_results(self) -> None:
        candidates = [(0, 0.9), (1, 0.89), (2, 0.5)]
        vectors = [[1.0, 0.0, 0.0], [0.999, 0.001, 0.0], [0.0, 0.0, 1.0]]
        selected = _mmr_select(candidates, vectors, limit=2, lambda_=0.3)
        self.assertEqual(selected, [0, 2])


class IndexMigrationTests(unittest.TestCase):
    def test_changed_embedder_triggers_reindex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = KnowledgeBase(
                HashingEmbedder(64), chunk_size=80, overlap=10, index_dir=directory
            )
            asyncio.run(first.ingest("星火项目每周三晚上部署。", source="a.md"))

            second = KnowledgeBase(
                HashingEmbedder(128), chunk_size=80, overlap=10, index_dir=directory
            )
            stats = second.stats()
            self.assertGreater(stats["missing_vectors"], 0)
            self.assertIsNone(stats["dims"])

            hits = asyncio.run(second.search_hits("部署时间", top_k=1))
            self.assertTrue(hits)
            self.assertEqual(second.stats()["dims"], 128)
            self.assertEqual(second.stats()["missing_vectors"], 0)


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous_workspace = workspace_root()
        configure_policy(workspace=self.root)
        configure_knowledge_base(
            HashingEmbedder(128),
            chunk_size=80,
            overlap=10,
            index_dir=self.root / "knowledge",
        )

    def tearDown(self) -> None:
        reset_knowledge_base()
        configure_policy(workspace=self.previous_workspace)

    def test_ingest_search_list_delete_stats(self) -> None:
        notes = self.root / "notes.md"
        notes.write_text("项目代号是星火。部署窗口在每周三晚上。", encoding="utf-8")

        message = knowledge_ingest("notes.md")
        self.assertIn("摄入", message)
        result = knowledge_search("项目代号", top_k=2)
        self.assertIn("星火", result)

        listing = knowledge_list()
        self.assertIn("notes.md", listing)
        stats = knowledge_stats()
        self.assertIn("HashingEmbedder", stats)
        self.assertIn("index.jsonl", stats)

        deleted = knowledge_delete("notes.md")
        self.assertIn("已删除", deleted)
        self.assertIn("知识库为空", knowledge_search("项目代号", top_k=2))

    def test_directory_ingest_skips_binary_and_empty_files(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("星火项目部署窗口是周三。", encoding="utf-8")
        (docs / "b.txt").write_text("回滚步骤：切回上一个镜像。", encoding="utf-8")
        (docs / "empty.md").write_text("   \n", encoding="utf-8")
        (docs / "binary.bin").write_bytes(b"\x00\x01\x02")
        (docs / "ignored.exe").write_bytes(b"MZ")

        message = knowledge_ingest_dir("docs", recursive=False)
        self.assertIn("2/3", message)
        self.assertIn("empty.md", message)
        listing = knowledge_list()
        self.assertIn("docs/a.md", listing)
        self.assertIn("docs/b.txt", listing)
        self.assertNotIn("binary.bin", listing)

    def test_directory_ingest_recursive(self) -> None:
        nested = self.root / "docs" / "deep"
        nested.mkdir(parents=True)
        (nested / "c.md").write_text("嵌套目录里的部署说明。", encoding="utf-8")
        message = knowledge_ingest_dir("docs", recursive=True)
        self.assertIn("1/1", message)
        self.assertIn("deep/c.md", knowledge_list())


if __name__ == "__main__":
    unittest.main()
