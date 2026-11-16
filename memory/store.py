"""长期记忆存储：每个会话一份 JSONL，跨进程重启保留。

与 session/ 的对话历史不同，这里只存"值得长期记住"的条目：
- 按 session_key 隔离，不同用户/渠道之间不会串记忆；
- 每条记忆有 id，模型可精确删除某一条；
- 单进程内用锁串行化读写，写入走临时文件 + os.replace；
- 读取时跳过损坏行；容量有硬上限，超出后淘汰最旧的条目。

多进程部署时不要共享同一个 memory_dir，否则文件级并发无法保证。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..utils.naming import safe_key

logger = logging.getLogger(__name__)

MAX_ENTRY_CHARS = 2_000
_MAX_SEARCH_RESULTS = 20
_CORRUPTION_WARN_LIMIT = 3
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_EN_TOKEN_RE = re.compile(r"[0-9a-z_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class MemoryEntry:
    """一条长期记忆。"""

    id: str
    text: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            created_at=str(data["created_at"]),
        )


def _prepare_directory(path: str) -> None:
    existed = os.path.isdir(path)
    os.makedirs(path, exist_ok=True, mode=0o700)
    if os.name != "nt" and not existed:
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def normalize_text(text: str) -> str:
    """压成单行、去掉控制字符，避免一条记忆把提示词结构撑坏。"""
    cleaned = _CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def render_entries(entries: list[MemoryEntry], max_chars: int) -> str:
    """渲染给提示词用的记忆块：新的在前，超出预算时保留最新条目。"""
    if max_chars <= 0 or not entries:
        return ""
    lines: list[str] = []
    used = 0
    for entry in reversed(entries):
        line = f"- [{entry.id}] {entry.created_at[:10]} {entry.text}"
        if used + len(line) + 1 > max_chars:
            if not lines:
                lines.append(line[: max_chars - 1] + "…")
            break
        lines.append(line)
        used += len(line) + 1
    omitted = len(entries) - len(lines)
    if omitted > 0:
        lines.append(f"- …另有 {omitted} 条较早记忆未展示，可用 memory_search 检索")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _query_terms(query: str) -> tuple[str, set[str]]:
    """提取英文词与中文二元组，做轻量的关键词检索（不依赖嵌入服务）。"""
    lowered = query.strip().lower()
    terms = {token for token in _EN_TOKEN_RE.findall(lowered) if len(token) >= 2}
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) <= 2:
            terms.add(run)
        else:
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return lowered, terms


class MemoryBackend(Protocol):
    """长期记忆后端的统一接口；本地 JSONL 与 Mem0 都实现它。"""

    enabled: bool

    def entries(self, key: str) -> list[MemoryEntry]: ...

    def count(self, key: str) -> int: ...

    def search(self, key: str, query: str, limit: int = 5) -> list[MemoryEntry]: ...

    def render(self, key: str, max_chars: int, query: str = "") -> str: ...

    def add(self, key: str, text: str) -> tuple[MemoryEntry, bool]: ...

    def remove(self, key: str, entry_id: str) -> bool: ...

    def clear(self, key: str) -> int: ...


class MemoryStore:
    """按 session_key 隔离的长期记忆仓库。"""

    def __init__(self, directory: str, max_entries: int = 200, enabled: bool = True):
        if not directory:
            raise ValueError("memory_dir 不能为空")
        if max_entries < 1:
            raise ValueError("memory_max_entries 必须 >= 1")
        self.directory = directory
        self.max_entries = max_entries
        self.enabled = enabled
        self._cache: dict[str, list[MemoryEntry]] = {}
        self._mtimes: dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def entries(self, key: str) -> list[MemoryEntry]:
        """返回某个会话的记忆（从旧到新）。文件被外部修改时会自动重读。"""
        if not self.enabled:
            return []
        with self._lock:
            return list(self._entries_locked(key))

    def count(self, key: str) -> int:
        return len(self.entries(key))

    def search(self, key: str, query: str, limit: int = 5) -> list[MemoryEntry]:
        """关键词检索，返回最相关且最新的若干条。"""
        lowered, terms = _query_terms(query)
        if not lowered:
            return []
        limit = max(1, min(int(limit), _MAX_SEARCH_RESULTS))
        with self._lock:
            entries = self._entries_locked(key)
            scored: list[tuple[int, int, MemoryEntry]] = []
            for index, entry in enumerate(entries):
                text = entry.text.lower()
                score = 5 if lowered in text else 0
                for term in terms:
                    hits = text.count(term)
                    if hits:
                        score += min(hits, 3)
                if score:
                    scored.append((score, index, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:limit]]

    def render(self, key: str, max_chars: int, query: str = "") -> str:
        """渲染给提示词用的记忆块：新的在前，超出预算时保留最新条目。

        ``query`` 只为兼容统一接口；本地后端不做语义检索，仍注入最近记忆。
        """
        return render_entries(self.entries(key), max_chars)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def add(self, key: str, text: str) -> tuple[MemoryEntry, bool]:
        """新增一条记忆，返回 (条目, 是否已存在)。

        重复保存同一内容不会产生副本，只把该条刷新到最新位置。
        """
        if not self.enabled:
            raise RuntimeError("长期记忆已禁用（memory_enabled=false）")
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("记忆内容不能为空")
        if len(normalized) > MAX_ENTRY_CHARS:
            raise ValueError(f"单条记忆不能超过 {MAX_ENTRY_CHARS} 字符")

        with self._lock:
            entries = self._entries_locked(key)
            for entry in entries:
                if entry.text == normalized:
                    if entries[-1] is not entry:
                        entries.remove(entry)
                        entries.append(entry)
                        self._write_locked(key, entries)
                    return entry, True

            entry = MemoryEntry(
                id=self._new_id(entries),
                text=normalized,
                created_at=_now(),
            )
            entries.append(entry)
            if len(entries) > self.max_entries:
                entries = entries[-self.max_entries :]
            self._write_locked(key, entries)
            return entry, False

    def remove(self, key: str, entry_id: str) -> bool:
        """按 id 删除；返回是否真的删掉了。"""
        if not self.enabled:
            return False
        with self._lock:
            entries = self._entries_locked(key)
            kept = [entry for entry in entries if entry.id != entry_id]
            if len(kept) == len(entries):
                return False
            self._write_locked(key, kept)
            return True

    def clear(self, key: str) -> int:
        """清空某个会话的全部长期记忆，返回删除条数。"""
        if not self.enabled:
            return 0
        with self._lock:
            entries = self._entries_locked(key)
            if not entries and not os.path.exists(self._path(key)):
                return 0
            self._write_locked(key, [])
            return len(entries)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{safe_key(key)}.jsonl")

    @staticmethod
    def _mtime(path: str) -> int:
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return 0

    def _entries_locked(self, key: str) -> list[MemoryEntry]:
        path = self._path(key)
        mtime = self._mtime(path)
        if key not in self._cache or self._mtimes.get(key) != mtime:
            self._load_locked(key, path, mtime)
        return self._cache[key]

    def _load_locked(self, key: str, path: str, mtime: int) -> None:
        entries: list[MemoryEntry] = []
        bad_lines = 0
        try:
            with open(path, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(MemoryEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        bad_lines += 1
                        if bad_lines <= _CORRUPTION_WARN_LIMIT:
                            logger.warning(
                                "记忆文件 %s 第 %d 行损坏，已跳过: %s",
                                path,
                                line_number,
                                exc,
                            )
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("无法读取记忆文件 %s: %s", path, exc)
        if bad_lines > _CORRUPTION_WARN_LIMIT:
            logger.warning("记忆文件 %s 共有 %d 行损坏，将在下次写入时清理", path, bad_lines)
        if len(entries) > self.max_entries:
            entries = entries[-self.max_entries :]
        self._cache[key] = entries
        self._mtimes[key] = mtime

    def _write_locked(self, key: str, entries: list[MemoryEntry]) -> None:
        path = self._path(key)
        _prepare_directory(self.directory)
        lines = [json.dumps(entry.to_dict(), ensure_ascii=False) for entry in entries]
        fd, temp_path = tempfile.mkstemp(prefix=".memory-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if lines:
                    handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        self._cache[key] = list(entries)
        self._mtimes[key] = self._mtime(path)

    @staticmethod
    def _new_id(entries: list[MemoryEntry]) -> str:
        existing = {entry.id for entry in entries}
        while True:
            entry_id = uuid.uuid4().hex[:12]
            if entry_id not in existing:
                return entry_id


_STORE: MemoryBackend | None = None


def configure_memory_store(store: MemoryBackend | None) -> MemoryBackend | None:
    """设置进程级长期记忆仓库；MiniBot 装配时调用。"""
    global _STORE
    _STORE = store
    return store


def get_memory_store() -> MemoryBackend | None:
    return _STORE
