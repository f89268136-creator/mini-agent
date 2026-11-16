"""把"当前正在处理哪个会话"传给工具线程。

工具函数只拿得到模型给的参数，拿不到 AgentLoop 的 session_key。
这里用 ContextVar 在每次处理消息时设置作用域；``asyncio.to_thread`` 会复制
执行上下文，因此记忆工具在工作线程里仍能读到正确的会话键。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_current_scope: ContextVar[str | None] = ContextVar(
    "mini_agent_memory_scope", default=None
)


def current_memory_scope() -> str | None:
    return _current_scope.get()


@contextmanager
def memory_scope(session_key: str) -> Iterator[None]:
    token = _current_scope.set(session_key)
    try:
        yield
    finally:
        _current_scope.reset(token)
