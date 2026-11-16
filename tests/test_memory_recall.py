"""长期记忆自动召回回归测试：按当前问题合并 Top-K 与最近记忆。

全部离线运行（不需要安装 mem0ai）：
    python -m unittest mini_agent.tests.test_memory_recall -v
"""

from __future__ import annotations

import tempfile
import unittest

from mini_agent.memory.mem0_store import Mem0Store
from mini_agent.memory.store import MemoryEntry, MemoryStore


def _entry(entry_id: str, text: str) -> MemoryEntry:
    return MemoryEntry(id=entry_id, text=text, created_at="2026-11-01T00:00:00+08:00")


def _fake_store(recent, hits=None, search_error=None):
    """绕过 Mem0 初始化，只验证 render 的合并策略。"""
    calls: list[tuple[str, str, int]] = []
    store = object.__new__(Mem0Store)
    store.enabled = True
    store.entries = lambda key: list(recent)

    def search(key: str, query: str, limit: int = 5):
        calls.append((key, query, limit))
        if search_error is not None:
            raise search_error
        return list(hits or [])

    store.search = search
    return store, calls


class Mem0RenderTests(unittest.TestCase):
    def test_merges_top_k_with_recent_and_dedupes(self) -> None:
        recent = [_entry(f"r{index}", f"旧记忆{index}") for index in range(7)]
        hits = [recent[3], _entry("hit", "用户偏好暗色主题")]
        store, calls = _fake_store(recent, hits)

        block = store.render("sdk:u", 10_000, "界面主题怎么设置")

        self.assertEqual(calls, [("sdk:u", "界面主题怎么设置", 5)])
        # 最近记忆只取 5 条，最旧的两条不注入。
        self.assertNotIn("旧记忆0", block)
        self.assertNotIn("旧记忆1", block)
        self.assertIn("旧记忆6", block)
        # 召回命中与最近记忆重叠时只出现一次。
        self.assertEqual(block.count("旧记忆3"), 1)
        self.assertIn("用户偏好暗色主题", block)
        # render_entries 从末尾向前渲染：命中项应排在最近记忆前面，优先展示。
        self.assertLess(
            block.index("用户偏好暗色主题"), block.index("旧记忆6")
        )

    def test_search_failure_falls_back_to_recent(self) -> None:
        recent = [_entry("a", "记忆A"), _entry("b", "记忆B")]
        store, calls = _fake_store(recent, search_error=RuntimeError("down"))

        block = store.render("sdk:u", 10_000, "任意问题")

        self.assertEqual(len(calls), 1)
        self.assertIn("记忆A", block)
        self.assertIn("记忆B", block)

    def test_empty_query_skips_semantic_search(self) -> None:
        recent = [_entry("a", "记忆A")]
        store, calls = _fake_store(recent)

        block = store.render("sdk:u", 10_000, "   ")

        self.assertEqual(calls, [])
        self.assertIn("记忆A", block)

    def test_entries_failure_still_returns_hits(self) -> None:
        store = object.__new__(Mem0Store)
        store.enabled = True

        def broken(key: str):
            raise RuntimeError("down")

        store.entries = broken
        store.search = lambda key, query, limit=5: [_entry("h", "命中的记忆")]

        block = store.render("sdk:u", 10_000, "问题")

        self.assertIn("命中的记忆", block)

    def test_disabled_or_zero_budget_returns_empty(self) -> None:
        store, calls = _fake_store([_entry("a", "记忆A")])
        self.assertEqual(store.render("sdk:u", 0, "问题"), "")
        store.enabled = False
        self.assertEqual(store.render("sdk:u", 10_000, "问题"), "")
        self.assertEqual(calls, [])


class LocalMemoryRenderTests(unittest.TestCase):
    def test_local_render_ignores_query_and_keeps_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.add("sdk:u", "用户使用 Python 写后端")
            store.add("sdk:u", "用户偏好简洁回答")

            block = store.render("sdk:u", 10_000, "完全不相关的问题")

            self.assertIn("用户使用 Python 写后端", block)
            self.assertIn("用户偏好简洁回答", block)


if __name__ == "__main__":
    unittest.main()