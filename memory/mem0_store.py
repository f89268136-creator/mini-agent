"""Mem0 记忆后端：用 LLM 抽取事实，并把向量写进本地 Qdrant。

设计约束：
- 只在 ``memory_backend=mem0`` 时由工厂实例化，未安装 mem0ai 不影响默认路径；
- 复用项目已有的 OpenAI 兼容端点，不额外要求 Mem0 云账号；
- 仍按 session_key 隔离：映射为 Mem0 的 user_id，agent_id 固定为 mini_agent；
- Mem0 的同步调用可能阻塞，调用方（ContextBuilder）应放到线程里执行。
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from typing import Any

from .store import MAX_ENTRY_CHARS, MemoryEntry, normalize_text, render_entries

logger = logging.getLogger(__name__)

_AGENT_ID = "mini_agent"
_COLLECTION = "mini_agent_memory"
# 自动注入：按当前问题取最相关若干条，再补最近若干条，避免只看到旧事实。
_AUTO_RECALL_LIMIT = 5
_AUTO_RECENT_LIMIT = 5


class Mem0Store:
    """把 Mem0 的同步 API 适配成项目的 MemoryBackend 接口。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        api_base: str,
        embedding_model: str | None,
        embedding_dims: int,
        directory: str,
        max_entries: int = 200,
        enabled: bool = True,
    ) -> None:
        if not directory:
            raise ValueError("mem0_dir 不能为空")
        if max_entries < 1:
            raise ValueError("memory_max_entries 必须 >= 1")
        self.directory = directory
        self.max_entries = max_entries
        self.enabled = enabled
        self._lock = threading.RLock()
        self._memory: Any = None
        if not enabled:
            return

        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError(
                "memory_backend=mem0 需要安装 mem0ai：pip install 'mem0ai>=2.2,<3'"
            ) from exc

        os.makedirs(directory, exist_ok=True)
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": model,
                    "api_key": api_key or "EMPTY",
                    "openai_base_url": api_base,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model or "text-embedding-3-small",
                    "api_key": api_key or "EMPTY",
                    "openai_base_url": api_base,
                    # 不向 API 传 dimensions：部分兼容端点不支持该参数；
                    # 维度只用于 Qdrant 集合（必须与模型实际输出一致）。
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": _COLLECTION,
                    "path": os.path.abspath(os.path.join(directory, "qdrant")),
                    "embedding_model_dims": embedding_dims,
                    "on_disk": True,
                },
            },
            "history_db_path": os.path.abspath(os.path.join(directory, "history.db")),
        }
        try:
            self._memory = Memory.from_config(config)
        except Exception as exc:
            raise RuntimeError(f"Mem0 初始化失败：{exc}") from exc

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def entries(self, key: str) -> list[MemoryEntry]:
        if not self.enabled:
            return []
        items = self._call(
            "读取记忆", self._memory.get_all, filters=self._filters(key), top_k=self.max_entries
        )
        entries = [entry for entry in map(self._to_entry, self._results(items)) if entry]
        entries.sort(key=lambda entry: entry.created_at)
        return entries

    def count(self, key: str) -> int:
        return len(self.entries(key))

    def search(self, key: str, query: str, limit: int = 5) -> list[MemoryEntry]:
        if not self.enabled or not query.strip():
            return []
        limit = max(1, min(int(limit), 20))
        items = self._call(
            "检索记忆",
            self._memory.search,
            query,
            filters=self._filters(key),
            top_k=limit,
        )
        return [entry for entry in map(self._to_entry, self._results(items)) if entry]

    def render(self, key: str, max_chars: int, query: str = "") -> str:
        """注入提示词：按当前问题召回 Top-K，再与最近记忆合并。

        检索失败只记日志并退回最近记忆；读取也失败时本轮跳过注入，
        绝不让记忆层影响主对话。
        """
        if not self.enabled or max_chars <= 0:
            return ""
        try:
            recent = self.entries(key)
        except Exception as exc:  # noqa: BLE001 - 注入失败不能中断整轮对话
            logger.warning("Mem0 最近记忆读取失败：%s", exc)
            recent = []

        hits: list[MemoryEntry] = []
        if query.strip():
            try:
                hits = self.search(key, query, limit=_AUTO_RECALL_LIMIT)
            except Exception as exc:  # noqa: BLE001 - 检索失败退回最近记忆
                logger.warning("Mem0 自动检索失败，退回最近记忆：%s", exc)

        if not recent and not hits:
            return ""
        # render_entries 从列表末尾向前渲染：把命中项放末尾，保证优先展示。
        merged: list[MemoryEntry] = []
        seen: set[str] = set()
        for entry in recent[-_AUTO_RECENT_LIMIT:]:
            if entry.id not in seen:
                seen.add(entry.id)
                merged.append(entry)
        for entry in hits:
            if entry.id not in seen:
                seen.add(entry.id)
                merged.append(entry)
        return render_entries(merged, max_chars)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def add(self, key: str, text: str) -> tuple[MemoryEntry, bool]:
        if not self.enabled:
            raise RuntimeError("长期记忆已禁用（memory_enabled=false）")
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("记忆内容不能为空")
        if len(normalized) > MAX_ENTRY_CHARS:
            raise ValueError(f"单条记忆不能超过 {MAX_ENTRY_CHARS} 字符")

        response = self._call(
            "保存记忆",
            self._memory.add,
            [{"role": "user", "content": normalized}],
            user_id=key,
            agent_id=_AGENT_ID,
            infer=True,
        )
        items = self._results(response)
        entry = next(
            (entry for entry in map(self._to_entry, items) if entry),
            None,
        )
        if entry is None:
            existing = self._find_existing(key, normalized)
            if existing is not None:
                return existing, True
            raise ValueError("Mem0 没有从内容中提取到可保存的事实")
        duplicate = bool(items) and all(
            str(item.get("event", "")).upper() == "NONE" for item in items
        )
        return entry, duplicate

    def remove(self, key: str, entry_id: str) -> bool:
        if not self.enabled:
            return False
        entry_id = entry_id.strip()
        if not entry_id:
            return False
        if not any(entry.id == entry_id for entry in self.entries(key)):
            return False
        self._call("删除记忆", self._memory.delete, entry_id)
        return True

    def clear(self, key: str) -> int:
        if not self.enabled:
            return 0
        count = self.count(key)
        if not count:
            return 0
        self._call(
            "清空记忆", self._memory.delete_all, user_id=key, agent_id=_AGENT_ID
        )
        return count

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    @staticmethod
    def _filters(key: str) -> dict[str, str]:
        return {"user_id": key, "agent_id": _AGENT_ID}

    @staticmethod
    def _results(response: Any) -> list[dict[str, Any]]:
        items = response.get("results", []) if isinstance(response, dict) else response
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _to_entry(item: dict[str, Any]) -> MemoryEntry | None:
        text = str(item.get("memory") or "").strip()
        entry_id = str(item.get("id") or "").strip()
        if not text or not entry_id:
            return None
        created_at = str(
            item.get("created_at") or item.get("updated_at") or ""
        ).strip()
        if not created_at:
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return MemoryEntry(id=entry_id, text=text, created_at=created_at)

    def _find_existing(self, key: str, text: str) -> MemoryEntry | None:
        for entry in self.entries(key):
            if entry.text == text:
                return entry
        return None

    def _call(self, action: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            with self._lock:
                return func(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Mem0 {action}失败：{exc}") from exc
