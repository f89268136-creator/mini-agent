"""当前请求上下文：让工具知道自己在替哪个渠道、哪个会话工作。

为什么要单独一层：
- ``spawn`` / ``delegate`` 需要知道父会话键、回复渠道和发送者，才能把后台
  结果送回同一条会话；
- 这些信息不能从模型参数里接收，否则模型可以伪造会话键去读取别人的记忆或
  往别的渠道发消息；
- 与 ``memory_scope`` 一样使用 ``ContextVar``：``AgentLoop`` 在每轮对话前
  设置，工具在协程或 ``asyncio.to_thread`` 的工作线程里都能安全读到。

后台任务（``asyncio.create_task``）会复制创建时的上下文，因此子代理任务
天然继承父会话的元数据；不需要也不允许把 session_key 暴露给模型。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterator

ApprovalCallback = Callable[[str, dict], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """一条入站消息在处理期间的路由元数据。"""

    channel: str
    sender: str
    session_key: str
    # 后台结果能否通过总线送回：需要入站/出站分发器正在运行，且渠道已装配。
    background_ready: bool = False
    # 有副作用工具的执行确认；None 表示当前渠道没有确认能力，按拒绝处理。
    approval: ApprovalCallback | None = None


_current: ContextVar[RequestContext | None] = ContextVar(
    "mini_agent_request_context", default=None
)


def current_request_context() -> RequestContext | None:
    """返回当前正在处理的请求；脱离 AgentLoop 调用工具时为 None。"""
    return _current.get()


@contextmanager
def request_scope(context: RequestContext) -> Iterator[None]:
    """在 with 块内绑定请求上下文，退出时恢复上一层。"""
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
