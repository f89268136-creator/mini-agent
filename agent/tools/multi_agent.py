"""多智能体工具：delegate（等结果）与 spawn（后台派生）。

两个工具都是协程函数：子代理本身要 await 模型与工具，不能放进线程池。
``concurrency_safe=True`` 让同一轮里的多个 delegate 并发执行，形成 fan-out
（例如"同时调研三个方向"），结果仍按模型给出的调用顺序回灌。

父会话元数据来自 :mod:`..request` 的 ContextVar，模型无法伪造 session_key。
"""

from __future__ import annotations

from ..request import current_request_context
from ..subagent import get_subagent_manager
from .policy import multi_agent_enabled
from .registry import register_tool


def _available() -> bool:
    return multi_agent_enabled()


def _context_error() -> str:
    if get_subagent_manager() is None:
        return "多智能体未配置（只有 MiniBot 会装配子代理管理器）。"
    if current_request_context() is None:
        return "无法确定当前会话，子代理调用被拒绝。"
    return ""


@register_tool(available=_available, concurrency_safe=True)
async def delegate(task: str, label: str = "") -> str:
    """把一个自包含的子任务交给新的子代理执行并等待结果。

    子代理有独立上下文与普通工具，无法向用户追问；适合需要独立完成的子任务。
    同一轮里多次调用 delegate 会并发执行。
    """
    error = _context_error()
    if error:
        return error
    if not task.strip():
        return "delegate 失败：task 不能为空。"
    manager = get_subagent_manager()
    context = current_request_context()
    assert manager is not None and context is not None
    return await manager.delegate(
        task.strip(),
        label=label.strip() or None,
        session_key=context.session_key,
    )


@register_tool(available=_available, concurrency_safe=True)
async def spawn(task: str, label: str = "") -> str:
    """在后台启动子代理并立即返回；完成后结果会自动发回当前会话。

    适合耗时较长、不需要当前轮次等待的独立任务。仅在 MiniBot 已启动渠道时可用；
    SDK 直接调用 chat() 时请改用 delegate。
    """
    error = _context_error()
    if error:
        return error
    if not task.strip():
        return "spawn 失败：task 不能为空。"
    context = current_request_context()
    manager = get_subagent_manager()
    assert manager is not None and context is not None
    if not context.background_ready:
        return (
            "spawn 不可用：后台结果需要正在运行的渠道。"
            "当前模式下请使用 delegate 等待结果。"
        )
    return await manager.spawn(
        task.strip(),
        label=label.strip() or None,
        session_key=context.session_key,
        channel=context.channel,
        sender=context.sender,
    )
