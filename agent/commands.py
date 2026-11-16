"""可注册的斜杠命令框架。

内置命令（/help、/reset、/status…）与外部注册的命令共用同一张表：
AgentLoop 只负责解析 "/名称 参数"，具体行为由注册的 async handler 决定。
这样新增命令不必再改 ``_handle_text`` 里的 if 链。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandContext:
    """一次命令调用的上下文；处理函数不应修改其中字段。"""

    channel: str
    sender: str
    session_key: str
    args: str = ""


CommandHandler = Callable[[CommandContext], Awaitable[str]]


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    usage: str
    handler: CommandHandler


@dataclass
class CommandRegistry:
    """命令名 -> 处理函数；支持别名，后注册的命令会覆盖同名的旧命令。"""

    _commands: dict[str, Command] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
        usage: str = "",
        aliases: tuple[str, ...] = (),
    ) -> Command:
        canonical = self._clean_name(name)
        command = Command(
            name=canonical,
            description=description,
            usage=usage or f"/{canonical}",
            handler=handler,
        )
        self._commands[canonical] = command
        for alias in aliases:
            alias = self._clean_name(alias)
            if alias != canonical:
                self._aliases[alias] = canonical
        return command

    def get(self, name: str) -> Command | None:
        canonical = self._aliases.get(name, name)
        return self._commands.get(canonical)

    def parse(self, text: str) -> tuple[str, str] | None:
        """把 "/name 参数" 解析成 (name, 参数)；不是命令时返回 None。"""
        body = text.strip()
        if not body.startswith("/"):
            return None
        body = body[1:].strip()
        if not body:
            return None
        name, _, args = body.partition(" ")
        return name, args.strip()

    def help_text(self) -> str:
        lines = ["可用命令："]
        for command in self._commands.values():
            lines.append(f"  {command.usage:<22}{command.description}")
        return "\n".join(lines)

    @staticmethod
    def _clean_name(name: str) -> str:
        value = name.strip().lstrip("/")
        if not value or any(ch.isspace() or ch == "/" for ch in value):
            raise ValueError(f"非法命令名: {name!r}")
        return value
