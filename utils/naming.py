"""把 session_key 转成安全且互不碰撞的文件名。

会话历史、长期记忆与任务清单共用同一套命名规则，保证同一个 key 总能映射到
同一组文件。文件名由“可读前缀 + 原始键的短哈希”组成，不同 key 不会共享文件。
"""

from __future__ import annotations

import hashlib
import re

_SAFE_KEY_RE = re.compile(r"[^0-9A-Za-z]")
_MAX_PREFIX_CHARS = 48


def safe_key(key: str) -> str:
    """可读前缀 + 短哈希（`cli:local` -> `cli_local-9f2c1b0a44`）。"""
    prefix = _SAFE_KEY_RE.sub("_", key).strip("_")[:_MAX_PREFIX_CHARS] or "key"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"
