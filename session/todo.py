"""会话级任务清单：让模型在复杂任务里显式跟踪进度。

- 按 session_key 隔离，一个会话一个 JSON 文件，进程重启后仍然存在；
- 状态只有 pending / in_progress / completed，同一时刻最多一项进行中；
- 模型通过 todo_write 整体替换清单，通过 todo_read 回看；
- 写入走临时文件 + os.replace，读取时跳过损坏文件并降级为空清单。

任务清单是"工作记忆"，不是长期记忆：它只描述当前会话正在做的事，
所以和 memory/ 分开存放，互不影响。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime

from ..utils.naming import safe_key

logger = logging.getLogger(__name__)

VALID_STATUSES = ("pending", "in_progress", "completed")
MAX_CONTENT_CHARS = 500
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_STATUS_MARK = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
_STATUS_ALIASES = {
    "pending": "pending",
    "todo": "pending",
    "open": "pending",
    "待办": "pending",
    "in_progress": "in_progress",
    "in-progress": "in_progress",
    "doing": "in_progress",
    "active": "in_progress",
    "进行中": "in_progress",
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "finished": "completed",
    "已完成": "completed",
    "完成": "completed",
}


@dataclass(frozen=True)
class TodoItem:
    content: str
    status: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content,
            "status": self.status,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TodoItem":
        return cls(
            content=str(data["content"]),
            status=str(data["status"]),
            updated_at=str(data["updated_at"]),
        )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_content(text: str) -> str:
    cleaned = _CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _prepare_directory(path: str) -> None:
    existed = os.path.isdir(path)
    os.makedirs(path, exist_ok=True, mode=0o700)
    if os.name != "nt" and not existed:
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


class TodoStore:
    """按 session_key 隔离的任务清单仓库。"""

    def __init__(self, directory: str, max_items: int = 50, enabled: bool = True):
        if not directory:
            raise ValueError("todos_dir 不能为空")
        if max_items < 1:
            raise ValueError("todo_max_items 必须 >= 1")
        self.directory = directory
        self.max_items = max_items
        self.enabled = enabled
        self._cache: dict[str, list[TodoItem]] = {}
        self._mtimes: dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def items(self, key: str) -> list[TodoItem]:
        if not self.enabled:
            return []
        with self._lock:
            return list(self._items_locked(key))

    def render(self, key: str, max_chars: int = 4_000) -> str:
        """渲染成提示词/工具结果里的清单，超出预算时保留前面的任务。"""
        if max_chars <= 0:
            return ""
        items = self.items(key)
        if not items:
            return ""
        total = len(items)
        done = sum(1 for item in items if item.status == "completed")
        active = sum(1 for item in items if item.status == "in_progress")
        header = f"进度 {done}/{total} 已完成"
        if active:
            header += f"，{active} 项进行中"
        lines = [header]
        used = len(header) + 1
        for item in items:
            line = f"- {_STATUS_MARK.get(item.status, '[ ]')} {item.content}"
            if used + len(line) + 1 > max_chars:
                lines.append("- …（更多任务未展示）")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def write(self, key: str, raw_items) -> list[TodoItem]:
        """整体替换清单；返回规范化后的条目。非法输入抛 ValueError。"""
        if not self.enabled:
            raise RuntimeError("任务清单已禁用（todos_enabled=false）")
        items = self._parse(raw_items)
        with self._lock:
            self._write_locked(key, items)
        return list(items)

    def clear(self, key: str) -> int:
        """清空清单，返回删除的条目数。"""
        if not self.enabled:
            return 0
        with self._lock:
            items = self._items_locked(key)
            path = self._path(key)
            if not items and not os.path.exists(path):
                return 0
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("删除任务清单 %s 失败: %s", path, exc)
                return 0
            self._cache[key] = []
            self._mtimes[key] = 0
            return len(items)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _parse(self, raw_items) -> list[TodoItem]:
        if isinstance(raw_items, str):
            try:
                raw_items = json.loads(raw_items)
            except json.JSONDecodeError as exc:
                raise ValueError(f"todos 不是合法 JSON：{exc.msg}") from exc
        if not isinstance(raw_items, list):
            raise ValueError("todos 必须是数组")
        if len(raw_items) > self.max_items:
            raise ValueError(
                f"任务清单最多 {self.max_items} 项，当前 {len(raw_items)} 项"
            )

        items: list[TodoItem] = []
        in_progress = 0
        for index, raw in enumerate(raw_items, start=1):
            if isinstance(raw, str):
                content, status = raw, "pending"
            elif isinstance(raw, dict):
                content = raw.get("content") or raw.get("title") or raw.get("task")
                status = raw.get("status", "pending")
            else:
                raise ValueError(f"第 {index} 项必须是对象或字符串")
            if content is None or not str(content).strip():
                raise ValueError(f"第 {index} 项缺少 content")
            content = _normalize_content(str(content))
            if not content:
                raise ValueError(f"第 {index} 项的 content 不能为空")
            if len(content) > MAX_CONTENT_CHARS:
                raise ValueError(
                    f"第 {index} 项超过 {MAX_CONTENT_CHARS} 字符"
                )
            normalized = _STATUS_ALIASES.get(str(status).strip().lower())
            if normalized is None:
                raise ValueError(
                    f"第 {index} 项状态无效：{status!r}；"
                    f"可选 {', '.join(VALID_STATUSES)}"
                )
            if normalized == "in_progress":
                in_progress += 1
                if in_progress > 1:
                    raise ValueError("同一时间只能有一项 in_progress")
            items.append(
                TodoItem(content=content, status=normalized, updated_at=_now())
            )
        return items

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{safe_key(key)}.json")

    @staticmethod
    def _mtime(path: str) -> int:
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return 0

    def _items_locked(self, key: str) -> list[TodoItem]:
        path = self._path(key)
        mtime = self._mtime(path)
        if key not in self._cache or self._mtimes.get(key) != mtime:
            self._load_locked(key, path, mtime)
        return self._cache[key]

    def _load_locked(self, key: str, path: str, mtime: int) -> None:
        items: list[TodoItem] = []
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            raw_items = data.get("items", []) if isinstance(data, dict) else data
            if isinstance(raw_items, list):
                for raw in raw_items:
                    if not isinstance(raw, dict):
                        continue
                    try:
                        item = TodoItem.from_dict(raw)
                    except (KeyError, TypeError, ValueError):
                        continue
                    content = _normalize_content(item.content)
                    if (
                        not content
                        or len(content) > MAX_CONTENT_CHARS
                        or item.status not in VALID_STATUSES
                    ):
                        continue
                    items.append(
                        TodoItem(
                            content=content,
                            status=item.status,
                            updated_at=item.updated_at,
                        )
                    )
            if len(items) > self.max_items:
                items = items[: self.max_items]
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("无法读取任务清单 %s，已按空清单处理: %s", path, exc)
            items = []
        self._cache[key] = items
        self._mtimes[key] = mtime

    def _write_locked(self, key: str, items: list[TodoItem]) -> None:
        path = self._path(key)
        _prepare_directory(self.directory)
        payload = {
            "version": 1,
            "items": [item.to_dict() for item in items],
        }
        fd, temp_path = tempfile.mkstemp(
            prefix=".todo-", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        self._cache[key] = list(items)
        self._mtimes[key] = self._mtime(path)


_STORE: TodoStore | None = None


def configure_todo_store(store: TodoStore | None) -> TodoStore | None:
    """设置进程级任务清单仓库；MiniBot 装配时调用。"""
    global _STORE
    _STORE = store
    return store


def get_todo_store() -> TodoStore | None:
    return _STORE
