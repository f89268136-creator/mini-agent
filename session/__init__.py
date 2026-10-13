"""会话功能包（对应 nanobot 的 session/）。

    manager.py  Session（一份按 key 隔离的历史）+ SessionManager（缓存 + JSONL 增量持久化）
"""

from .manager import Session, SessionManager

__all__ = ["Session", "SessionManager"]
