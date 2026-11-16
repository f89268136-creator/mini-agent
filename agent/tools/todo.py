"""任务清单工具：把复杂任务拆成可跟踪的步骤。

与长期记忆工具一样，会话键来自 ``current_memory_scope()``，模型无法指定
session_key，因此不能读写其他会话的清单。存储是同步文件 I/O，runner 会用
``asyncio.to_thread`` 执行，不会阻塞事件循环。
"""

from __future__ import annotations

from ...memory.scope import current_memory_scope
from ...session.todo import get_todo_store
from .registry import register_tool

_TODO_WRITE_PARAMETERS = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": (
                "按执行顺序排列的完整任务清单，会整体替换旧清单；"
                "传空数组表示清空"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "一句话描述这项任务，最多 500 字",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": (
                            "pending=未开始，in_progress=进行中（同时最多一项），"
                            "completed=已完成"
                        ),
                    },
                },
                "required": ["content", "status"],
            },
        }
    },
    "required": ["todos"],
}


def _todo_available() -> bool:
    store = get_todo_store()
    return store is not None and store.enabled


def _scope_error() -> str:
    store = get_todo_store()
    if store is None:
        return "任务清单未配置（只有 MiniBot 会装配任务清单）。"
    if not store.enabled:
        return "任务清单已禁用（todos_enabled=false）。"
    if not current_memory_scope():
        return "无法确定当前会话，任务清单操作被拒绝。"
    return ""


@register_tool(
    name="todo_write",
    parameters=_TODO_WRITE_PARAMETERS,
    available=_todo_available,
)
def todo_write(todos) -> str:
    """整体替换当前会话的任务清单。

    任务包含多个步骤、需要多次工具调用或跨越较长对话时，先用它建立清单；
    开始某项前把状态改为 in_progress，完成后立即改为 completed，保持清单
    与实际情况一致。简单的一步问答不要使用。
    """
    error = _scope_error()
    if error:
        return error
    store = get_todo_store()
    key = current_memory_scope()
    assert store is not None and key is not None
    try:
        items = store.write(key, todos)
    except (ValueError, RuntimeError) as exc:
        return f"任务清单更新失败：{exc}"
    if not items:
        return "任务清单已清空。"
    return f"任务清单已更新（共 {len(items)} 项）。\n{store.render(key)}"


@register_tool(available=_todo_available)
def todo_read() -> str:
    """查看当前会话的任务清单与完成进度；不确定还剩哪些步骤时调用。"""
    error = _scope_error()
    if error:
        return error
    store = get_todo_store()
    key = current_memory_scope()
    assert store is not None and key is not None
    rendered = store.render(key)
    if not rendered:
        return "当前会话还没有任务清单。复杂任务可以用 todo_write 建立清单。"
    return rendered
