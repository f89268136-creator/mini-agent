"""工具安全策略（进程级）。

工具注册表是进程级全局对象，因此策略也是进程级：一次进程只应服务
一套 ``workspace`` / ``allow_shell`` / ``allow_private_urls`` 配置。
默认值按生产安全优先：文件工具限制在当前工作目录，shell 与内网 URL 默认关闭。
"""

from __future__ import annotations

import os
from pathlib import Path

_workspace: Path = Path.cwd().resolve()
_allow_shell: bool = False
_allow_private_urls: bool = False
# ask：执行有副作用的工具前必须获得用户确认；off：信任当前调用方，不再询问。
_approval_mode: str = "ask"
# 默认关闭：只有 MiniBot/AgentLoop 装配了子代理管理器后才打开。
_multi_agent_enabled: bool = False


def configure_policy(
    *,
    workspace: str | os.PathLike[str] | None = None,
    allow_shell: bool | None = None,
    allow_private_urls: bool | None = None,
    tool_approval: str | None = None,
) -> None:
    """更新进程级工具策略；参数为 None 表示保持当前值。"""
    global _workspace, _allow_shell, _allow_private_urls, _approval_mode
    if workspace is not None:
        _workspace = Path(workspace).expanduser().resolve()
    if allow_shell is not None:
        _allow_shell = bool(allow_shell)
    if allow_private_urls is not None:
        _allow_private_urls = bool(allow_private_urls)
    if tool_approval is not None:
        mode = str(tool_approval).strip().lower()
        if mode not in ("ask", "off"):
            raise ValueError("tool_approval 只能是 'ask' 或 'off'")
        _approval_mode = mode


def configure_multi_agent(enabled: bool) -> None:
    """控制 delegate / spawn 是否出现在模型可见的工具列表里。"""
    global _multi_agent_enabled
    _multi_agent_enabled = bool(enabled)


def multi_agent_enabled() -> bool:
    return _multi_agent_enabled


def workspace_root() -> Path:
    return _workspace


def resolve_workspace_path(path: str, *, must_exist: bool = False) -> Path:
    """把用户/模型给出的路径解析到工作区内。

    相对路径基于 workspace；绝对路径必须位于 workspace 内。
    使用 ``resolve()`` 展开符号链接，防止通过链接逃逸出工作区。
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_workspace)
    except ValueError:
        raise PermissionError(
            f"路径越界：{path!r} 不在工作区 {str(_workspace)!r} 内"
        ) from None
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    return resolved


def shell_enabled() -> bool:
    return _allow_shell


def private_urls_enabled() -> bool:
    return _allow_private_urls


def approval_required(requires_approval: bool) -> bool:
    return bool(requires_approval) and _approval_mode == "ask"


def describe_policy() -> dict[str, object]:
    return {
        "workspace": str(_workspace),
        "allow_shell": _allow_shell,
        "allow_private_urls": _allow_private_urls,
        "tool_approval": _approval_mode,
    }
