"""长期记忆工具：模型显式保存 / 检索 / 删除跨会话事实。

这些工具依赖两个进程级状态：
- ``get_memory_store()``：MiniBot 装配时注入的存储；
- ``current_memory_scope()``：AgentLoop 在每轮对话前设置的 session_key。

工具在 ``asyncio.to_thread`` 的工作线程中执行，ContextVar 会随执行上下文复制，
因此不需要把 session_key 暴露给模型（避免模型越权读写别人的记忆）。
"""

from __future__ import annotations

from ...memory.scope import current_memory_scope
from ...memory.store import get_memory_store
from .registry import register_tool


def _memory_available() -> bool:
    store = get_memory_store()
    return store is not None and store.enabled


def _scope_error() -> str:
    store = get_memory_store()
    if store is None:
        return "长期记忆未配置（只有 MiniBot 会装配长期记忆）。"
    if not store.enabled:
        return "长期记忆已禁用（memory_enabled=false）。"
    if not current_memory_scope():
        return "无法确定当前会话，长期记忆操作被拒绝。"
    return ""


def _clamp_limit(value: int, default: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(number, maximum))


@register_tool(available=_memory_available)
def memory_save(text: str) -> str:
    """把值得长期记住的事实、用户偏好或约定写入当前会话；对话被清空后仍保留。"""
    error = _scope_error()
    if error:
        return error
    store = get_memory_store()
    key = current_memory_scope()
    assert store is not None and key is not None
    try:
        entry, duplicate = store.add(key, text)
    except (ValueError, RuntimeError) as exc:
        return f"保存失败：{exc}"
    if duplicate:
        return f"这条记忆已存在（id={entry.id}），未重复保存。"
    return f"已保存长期记忆（id={entry.id}，当前共 {store.count(key)} 条）。"


@register_tool(available=_memory_available)
def memory_search(query: str, limit: int = 5) -> str:
    """在当前会话的长期记忆里检索相关条目；用于回忆较早的偏好与约定。"""
    error = _scope_error()
    if error:
        return error
    store = get_memory_store()
    key = current_memory_scope()
    assert store is not None and key is not None
    hits = store.search(key, query, limit=_clamp_limit(limit, 5, 20))
    if not hits:
        return "没有检索到相关记忆。"
    lines = [f"找到 {len(hits)} 条相关记忆（新的在前）："]
    lines.extend(f"[{entry.id}] {entry.created_at[:10]} {entry.text}" for entry in hits)
    return "\n".join(lines)


@register_tool(available=_memory_available)
def memory_list(limit: int = 20) -> str:
    """列出当前会话最近的长期记忆条目，包含可用于删除的 id。"""
    error = _scope_error()
    if error:
        return error
    store = get_memory_store()
    key = current_memory_scope()
    assert store is not None and key is not None
    entries = store.entries(key)
    if not entries:
        return "当前会话还没有长期记忆。"
    shown = list(reversed(entries))[:_clamp_limit(limit, 20, 100)]
    lines = [f"长期记忆共 {len(entries)} 条，显示最近 {len(shown)} 条："]
    lines.extend(f"[{entry.id}] {entry.created_at[:10]} {entry.text}" for entry in shown)
    return "\n".join(lines)


@register_tool(available=_memory_available)
def memory_forget(entry_id: str) -> str:
    """按 id 删除当前会话的一条长期记忆；id 可从 memory_list / memory_search 获得。"""
    error = _scope_error()
    if error:
        return error
    store = get_memory_store()
    key = current_memory_scope()
    assert store is not None and key is not None
    if not entry_id.strip():
        return "entry_id 不能为空。"
    if store.remove(key, entry_id.strip()):
        return f"已删除长期记忆 {entry_id.strip()}。"
    return f"没有找到 id 为 {entry_id.strip()} 的记忆。"
