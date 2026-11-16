"""长期记忆：按会话隔离的 JSONL 存储、作用域与提示词注入。"""

from .factory import build_memory_backend
from .scope import current_memory_scope, memory_scope
from .store import (
    MAX_ENTRY_CHARS,
    MemoryBackend,
    MemoryEntry,
    MemoryStore,
    configure_memory_store,
    get_memory_store,
)

__all__ = [
    "MAX_ENTRY_CHARS",
    "MemoryBackend",
    "MemoryEntry",
    "MemoryStore",
    "build_memory_backend",
    "configure_memory_store",
    "get_memory_store",
    "current_memory_scope",
    "memory_scope",
]
