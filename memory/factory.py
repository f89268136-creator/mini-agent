"""长期记忆后端工厂：按配置选择本地 JSONL 或 Mem0。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .store import MemoryBackend, MemoryStore

if TYPE_CHECKING:
    from ..config import MiniConfig


def build_memory_backend(config: MiniConfig) -> MemoryBackend:
    """按 ``memory_backend`` 构造后端；Mem0 只在启用记忆时才导入。"""
    if config.memory_backend == "mem0" and config.memory_enabled:
        from .mem0_store import Mem0Store

        return Mem0Store(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            embedding_model=config.mem0_embedding_model or config.embedding_model,
            embedding_dims=config.mem0_embedding_dims,
            directory=config.mem0_dir,
            max_entries=config.memory_max_entries,
            enabled=True,
        )
    return MemoryStore(
        config.memory_dir,
        max_entries=config.memory_max_entries,
        enabled=config.memory_enabled,
    )
