"""会话仓库：按会话键隔离历史，JSONL 增量持久化。

持久化保证：
- 追加写：只写水位线之后的新消息；
- 历史收缩（/reset）或文件损坏时整文件原子重写（临时文件 + os.replace）；
- 写入后 flush + fsync，进程崩溃不会留下半条已确认的消息；
- 读取时跳过损坏行，并在下一次保存时自动重写干净文件。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

from ..providers.base import Message, ToolCall
from ..utils.naming import safe_key

logger = logging.getLogger(__name__)

_warned_open_dirs: set[str] = set()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _prepare_storage_dir(path: str) -> None:
    """创建会话目录；新建时收紧到 0700，已存在且过宽时只告警不擅自改。"""
    existed = os.path.isdir(path)
    os.makedirs(path, exist_ok=True, mode=0o700)
    if os.name == "nt":
        return
    try:
        if not existed:
            os.chmod(path, 0o700)
            return
        if os.stat(path).st_mode & 0o077 and path not in _warned_open_dirs:
            _warned_open_dirs.add(path)
            logger.warning(
                "会话目录 %s 对其他用户可读/可写，建议 chmod 700（对话内容可能含敏感信息）",
                path,
            )
    except OSError:
        return


def _restrict_new_file(path: str, created: bool) -> None:
    if created and os.name != "nt":
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)


def sanitize_tool_messages(messages: list[Message]) -> list[Message]:
    """丢弃不完整的工具调用链，避免把非法消息发给模型。

    OpenAI 兼容接口要求：assistant 的每个 tool_call 后面都必须有对应 id 的
    tool 消息。按条数截断历史时，调用和结果可能被切到两边；这里做一次
    配对检查——结果不全的 assistant 去掉 tool_calls（有正文则保留正文），
    找不到归属的 tool 消息直接丢弃。
    """
    result_positions: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        if message.role == "tool" and message.tool_call_id:
            result_positions.setdefault(message.tool_call_id, []).append(index)

    complete_assistants: set[int] = set()
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        if all(
            any(position > index for position in result_positions.get(call.id, []))
            for call in message.tool_calls
        ):
            complete_assistants.add(index)

    cleaned: list[Message] = []
    available: set[str] = set()
    fulfilled: set[str] = set()
    for index, message in enumerate(messages):
        if message.role == "tool":
            call_id = message.tool_call_id
            if call_id and call_id in available and call_id not in fulfilled:
                cleaned.append(message)
                fulfilled.add(call_id)
            continue
        if message.role == "assistant" and message.tool_calls:
            if index not in complete_assistants:
                if message.content:
                    cleaned.append(
                        Message(
                            role="assistant",
                            content=message.content,
                            metadata=message.metadata,
                        )
                    )
                continue
            available.update(call.id for call in message.tool_calls)
        cleaned.append(message)
    return cleaned


@dataclass
class Session:
    """一份会话历史。key 通常等于 InboundMessage.session_key。

    ``summary`` 是早期对话被压缩后的摘要，随 ``.meta.json`` 一起持久化；
    ``summary_messages`` 记录摘要覆盖过多少条原始消息，仅用于展示。
    """

    key: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    summary_messages: int = 0

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        """返回可以安全发给模型的消息快照。

        规则：
        1) 首条 system 永远保留；
        2) 只保留 system + 最近 max_messages 条；
        3) 清理不完整的 tool_calls / 孤儿 tool 消息。
        """
        messages = list(self.messages)
        system: list[Message] = []
        if messages and messages[0].role == "system":
            system = messages[:1]
            messages = messages[1:]
        if max_messages is not None:
            messages = messages[-max_messages:]
        return system + sanitize_tool_messages(messages)

    def reset(self) -> None:
        """清空对话，只留 system 消息；已压缩的摘要也一并丢弃。"""
        self.messages = [m for m in self.messages if m.role == "system"]
        self.summary = ""
        self.summary_messages = 0


def _message_from_dict(data: dict) -> Message:
    raw_calls = data.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=call["id"],
            name=call["function"]["name"],
            arguments=call["function"]["arguments"],
        )
        for call in raw_calls
    ]
    return Message(
        role=data["role"],
        content=data.get("content"),
        tool_calls=tool_calls or None,
        tool_call_id=data.get("tool_call_id"),
        metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
    )


class SessionManager:
    """会话的创建、缓存与增量持久化（单事件循环内使用）。"""

    def __init__(self, storage_dir: str, system_prompt: str, history_limit: int = 50) -> None:
        self.storage_dir = storage_dir
        self.system_prompt = system_prompt
        self.history_limit = history_limit
        self._cache: dict[str, Session] = {}
        self._saved_counts: dict[str, int] = {}
        self._save_locks: dict[str, asyncio.Lock] = {}
        # 文件里出现过损坏行 / 缺少 system：下次保存走整文件重写
        self._needs_rewrite: set[str] = set()

    def get_or_create(self, key: str) -> Session:
        """取会话：内存缓存 -> 磁盘恢复 -> 新建。"""
        if key in self._cache:
            return self._cache[key]

        session = self._load_from_disk(key)
        on_disk = len(session.messages)
        if not session.messages or session.messages[0].role != "system":
            session.messages.insert(0, Message(role="system", content=self.system_prompt))
            if on_disk > 0:
                # 文件存在但缺 system：内存补上，下次保存重写文件
                self._needs_rewrite.add(key)
                saved = len(session.messages)
            else:
                saved = 0
        else:
            saved = on_disk

        self._cache[key] = session
        self._saved_counts[key] = saved
        return session

    def _load_from_disk(self, key: str) -> Session:
        meta = self._read_meta(key)
        summary = str(meta.get("summary") or "")
        summary_messages = int(meta.get("summary_messages") or 0)
        path = self._path(key)
        if not os.path.exists(path):
            return Session(
                key=key, summary=summary, summary_messages=summary_messages
            )
        messages: list[Message] = []
        bad_lines = 0
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(_message_from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    bad_lines += 1
                    if bad_lines <= 3:
                        logger.warning(
                            "会话文件 %s 第 %d 行损坏，已跳过: %s",
                            path,
                            line_number,
                            exc,
                        )
        if bad_lines:
            logger.warning("会话文件 %s 共有 %d 行损坏，将在下次保存时重写", path, bad_lines)
            self._needs_rewrite.add(key)
        return Session(
            key=key,
            messages=messages,
            summary=summary,
            summary_messages=summary_messages,
        )

    async def save(self, session: Session) -> None:
        """增量保存；历史收缩或文件损坏时原子重写。"""
        async with self._lock_for(session.key):
            await self._save_locked(session)

    async def _save_locked(self, session: Session) -> None:
        _prepare_storage_dir(self.storage_dir)
        saved = self._saved_counts.get(session.key, 0)
        must_rewrite = session.key in self._needs_rewrite or len(session.messages) < saved

        if must_rewrite:
            lines = [
                json.dumps(m.to_dict(include_metadata=True), ensure_ascii=False)
                for m in session.messages
            ]
            await asyncio.to_thread(self._write_lines, self._path(session.key), lines)
            self._saved_counts[session.key] = len(session.messages)
            self._needs_rewrite.discard(session.key)
            await asyncio.to_thread(self._write_meta, session)
            return

        new_messages = session.messages[saved:]
        if not new_messages:
            await asyncio.to_thread(self._write_meta, session)
            return
        lines = [
            json.dumps(m.to_dict(include_metadata=True), ensure_ascii=False)
            for m in new_messages
        ]
        await asyncio.to_thread(self._append_lines, self._path(session.key), lines)
        self._saved_counts[session.key] = len(session.messages)
        await asyncio.to_thread(self._write_meta, session)

    async def evict(self, key: str) -> None:
        """把会话刷盘后移出内存缓存（空闲 worker 回收时调用）。"""
        async with self._lock_for(key):
            session = self._cache.get(key)
            if session is not None:
                await self._save_locked(session)
            self._cache.pop(key, None)
            self._saved_counts.pop(key, None)
            self._needs_rewrite.discard(key)

    def delete(self, key: str) -> None:
        """删除会话的正文与摘要文件。"""
        self._cache.pop(key, None)
        self._saved_counts.pop(key, None)
        self._needs_rewrite.discard(key)
        for path in (self._path(key), self._meta_path(key)):
            if os.path.exists(path):
                os.remove(path)

    def list_sessions(self, prefix: str = "") -> list[dict[str, object]]:
        """列出已落盘会话的摘要（标题、更新时间、消息数）。

        摘要来自 ``<safe_key>.meta.json`` 旁路文件：``safe_key`` 带哈希后缀，
        无法从文件名反推出原始会话键，所以旁路文件是服务端会话列表的唯一来源。
        只在内存里、尚未保存的会话也会被补进结果。
        """
        items: dict[str, dict[str, object]] = {}
        if os.path.isdir(self.storage_dir):
            for name in os.listdir(self.storage_dir):
                if not name.endswith(".meta.json"):
                    continue
                path = os.path.join(self.storage_dir, name)
                try:
                    with open(path, encoding="utf-8") as handle:
                        data = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                key = data.get("key")
                if not isinstance(key, str) or not key.startswith(prefix):
                    continue
                items[key] = {
                    "key": key,
                    "title": str(data.get("title") or "新对话"),
                    "updated_at": str(data.get("updated_at") or ""),
                    "messages": int(data.get("messages") or 0),
                }
        for key, session in self._cache.items():
            if not key.startswith(prefix) or key in items:
                continue
            items[key] = {
                "key": key,
                "title": self._session_title(session),
                "updated_at": "",
                "messages": len(session.messages),
            }
        return sorted(
            items.values(),
            key=lambda item: str(item["updated_at"]),
            reverse=True,
        )

    def _write_meta(self, session: Session) -> None:
        """原子写入会话摘要；失败只告警，不影响会话正文落盘。"""
        payload = {
            "key": session.key,
            "title": self._session_title(session),
            "updated_at": _now_iso(),
            "messages": len(session.messages),
            "summary": session.summary,
            "summary_messages": session.summary_messages,
        }
        path = self._meta_path(session.key)
        fd, temp_path = tempfile.mkstemp(
            prefix=".session-meta-", suffix=".tmp", dir=os.path.dirname(path) or "."
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except OSError:
            logger.warning("会话摘要写入失败: %s", path, exc_info=True)
        finally:
            if os.path.exists(temp_path):
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)

    @staticmethod
    def _session_title(session: Session) -> str:
        for message in session.messages:
            if message.role != "user" or not message.content:
                continue
            text = " ".join(message.content.split())
            if text:
                return text[:60] + ("…" if len(text) > 60 else "")
        return "新对话"

    def _meta_path(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"{safe_key(key)}.meta.json")

    def _read_meta(self, key: str) -> dict:
        try:
            with open(self._meta_path(key), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _append_lines(path: str, lines: list[str]) -> None:
        created = not os.path.exists(path)
        needs_newline = False
        if not created and os.path.getsize(path) > 0:
            with open(path, "rb") as handle:
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) != b"\n"
        with open(path, "a", encoding="utf-8") as handle:
            _restrict_new_file(path, created)
            if needs_newline:
                handle.write("\n")
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_lines(path: str, lines: list[str]) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if lines:
                    handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _path(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"{safe_key(key)}.jsonl")

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._save_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._save_locks[key] = lock
        return lock
