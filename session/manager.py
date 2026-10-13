"""部件 7：会话与持久化（对应 nanobot/session/manager.py 的最小版）。

一个 Session = 一份按 key 隔离的对话历史（记忆）。
SessionManager 负责两件事：
  - get_or_create：内存缓存优先，没有就尝试从磁盘恢复；
  - save：增量写盘——一条会话一个 JSONL 文件，只 append 新消息。

为什么这样设计？网关进程会重启，历史必须落盘；而"一条会话一个文件、
只追加新行"的写法最简单，也天然贴合"边聊边存"的场景。
nanobot 生产版是 JsonlSessionStore + 归档/压缩 + TTL 自动腾挪，
这里只做"保存-恢复"最小闭环。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

from ..providers.base import Message, ToolCall

# key（如 "cli:local"）不能直接当文件名，非字母数字统一替换为下划线
_SAFE_KEY_RE = re.compile(r"[^0-9A-Za-z]")


@dataclass
class Session:
    """一份会话历史。key 通常等于 InboundMessage.session_key。"""

    key: str
    messages: list[Message] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        """返回"可以发给模型"的消息列表。

        最小版清洗规则：
        1) 首条 system 永远保留（人设/规则不能被截掉）；
        2) 若给了 max_messages，只保留 system + 最近 max_messages 条
           （截断历史控制 token 成本的土办法）；
        3) 清理孤儿 tool 消息：tool 消息靠 tool_call_id 与前面 assistant
           消息里的 tool_call 配对；截断经常把"发起调用的 assistant"切掉，
           只剩结果，模型会因"无主的结果"报错，所以要丢掉孤儿；
        4) nanobot 还有更严的 role 交替清洗（user/assistant 严格交替等），
           这里刻意只做最小版。
        """
        messages = list(self.messages)
        system: list[Message] = []
        if messages and messages[0].role == "system":
            system = messages[:1]
            messages = messages[1:]
        if max_messages is not None:
            messages = messages[-max_messages:]

        # 按顺序扫描：assistant 的 tool_call id 先登记，tool 消息再认领
        call_ids: set[str] = set()
        cleaned: list[Message] = []
        for msg in messages:
            if msg.role == "tool":
                if msg.tool_call_id in call_ids:
                    cleaned.append(msg)
                # tool_call_id 找不到对应调用 => 孤儿，丢弃
                continue
            if msg.role == "assistant" and msg.tool_calls:
                call_ids.update(tc.id for tc in msg.tool_calls)
            cleaned.append(msg)
        return system + cleaned

    def reset(self) -> None:
        """清空对话，只留 system 消息（人设不丢）。"""
        self.messages = [m for m in self.messages if m.role == "system"]


def _safe_key(key: str) -> str:
    """把会话 key 清洗成安全文件名（"cli:local" -> "cli_local"）。"""
    return _SAFE_KEY_RE.sub("_", key)


def _message_from_dict(data: dict) -> Message:
    """把一行 JSONL 还原成 Message（to_dict 的逆操作）。"""
    raw_calls = data.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=c["id"],
            name=c["function"]["name"],
            arguments=c["function"]["arguments"],
        )
        for c in raw_calls
    ]
    return Message(
        role=data["role"],
        content=data.get("content"),
        tool_calls=tool_calls or None,
        tool_call_id=data.get("tool_call_id"),
    )


class SessionManager:
    """会话的创建、缓存与增量持久化。"""

    def __init__(self, storage_dir: str, system_prompt: str, history_limit: int = 50) -> None:
        self.storage_dir = storage_dir
        self.system_prompt = system_prompt
        self.history_limit = history_limit  # 供上层截断历史用，本类不强制
        self._cache: dict[str, Session] = {}
        # 增量保存的水位线：每个 key 已经写进磁盘的消息条数
        self._saved_counts: dict[str, int] = {}

    def get_or_create(self, key: str) -> Session:
        """取会话：内存缓存优先；没有就从磁盘恢复；都失败则新建。"""
        if key in self._cache:
            return self._cache[key]

        session = self._load_from_disk(key)
        on_disk = len(session.messages)  # 磁盘上已有的条数
        # 恢复出来的历史若缺少 system 消息（首次创建 / 文件被裁剪），补上人设
        if not session.messages or session.messages[0].role != "system":
            session.messages.insert(0, Message(role="system", content=self.system_prompt))
            # 空会话：补的 system 还没落盘，水位线从 0 开始，首次 save 会把它写进去；
            # 文件存在但缺 system（少见）：现有消息都算已保存，补的 system 只留内存
            saved = 0 if on_disk == 0 else len(session.messages)
        else:
            saved = on_disk

        self._cache[key] = session
        # 水位线 = 磁盘上已有的条数；之后 save 只 append 水位线之后的新消息
        self._saved_counts[key] = saved
        return session

    def _load_from_disk(self, key: str) -> Session:
        path = self._path(key)
        if not os.path.exists(path):
            return Session(key=key)
        messages: list[Message] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(_message_from_dict(json.loads(line)))
        return Session(key=key, messages=messages)

    async def save(self, session: Session) -> None:
        """增量持久化：只把水位线之后的新消息 append 到 JSONL 文件。

        每行一条 json.dumps(m.to_dict())，ensure_ascii=False 保证中文可读。
        """
        os.makedirs(self.storage_dir, exist_ok=True)
        saved = self._saved_counts.get(session.key, 0)

        # 历史收缩（比水位线还短）：只会发生在 /reset 清空之后——
        # 增量"追加"此刻语义失效（水位线高于当前长度，切片为空，
        # 之后的新消息还会错位落盘），必须整文件重写并重置水位线。
        if len(session.messages) < saved:
            lines = [
                json.dumps(m.to_dict(), ensure_ascii=False) for m in session.messages
            ]
            await asyncio.to_thread(self._write_lines, self._path(session.key), lines)
            self._saved_counts[session.key] = len(session.messages)
            return

        new_messages = session.messages[saved:]
        if not new_messages:
            return  # 没有新消息，不必碰磁盘

        lines = [json.dumps(m.to_dict(), ensure_ascii=False) for m in new_messages]
        # 写盘是阻塞 IO，丢进线程池执行，避免卡住事件循环
        await asyncio.to_thread(self._append_lines, self._path(session.key), lines)
        self._saved_counts[session.key] = len(session.messages)

    @staticmethod
    def _append_lines(path: str, lines: list[str]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _write_lines(path: str, lines: list[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    def _path(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"{_safe_key(key)}.jsonl")

    def list_keys(self) -> list[str]:
        """已知的会话 key（内存缓存中的原始 key）。

        注意：磁盘文件名经过 safe_key 清洗，可能撞名、无法无损还原，
        所以这里只报缓存里的 key；nanobot 生产版由存储层负责精确列出。
        """
        return sorted(self._cache.keys())
