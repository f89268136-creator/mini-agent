"""渠道包：渠道抽象 + 三个内置实现 + 工厂。

内置渠道通过显式映射装配；配置里出现未知渠道名会直接报错，
避免拼写错误导致渠道被静默跳过。自定义渠道可在 MiniBot 构造后
注入 ``bot.channels``，再调用 ``bot.start()``。

- ``cli``：终端交互；
- ``echo``：回环 TCP 行协议（演示用）；
- ``web``：回环 HTTP + SSE 的单页聊天前端。
"""

from __future__ import annotations

from ..bus import MessageBus
from .base import Channel
from .cli import CliChannel
from .echo import EchoChannel
from .web import WebChannel

__all__ = [
    "Channel",
    "CliChannel",
    "EchoChannel",
    "WebChannel",
    "build_channels",
    "validate_channels",
]

BUILTIN_CHANNEL_NAMES = ("cli", "echo", "web")


def validate_channels(names: list[str]) -> None:
    """校验渠道名，未知名字立即失败；不创建任何渠道实例。"""
    for name in names:
        if name not in BUILTIN_CHANNEL_NAMES:
            raise ValueError(
                f"未知渠道: {name!r}（可选: {', '.join(BUILTIN_CHANNEL_NAMES)}）"
            )


def build_channels(
    names: list[str],
    bus: MessageBus,
    *,
    echo_port: int = 8766,
    web_port: int = 8767,
) -> dict[str, Channel]:
    """按配置里的名字列表创建渠道实例，返回 {name: channel}。

    未知名字立即抛 ValueError（启动阶段失败优于运行期静默丢消息）。
    Web 渠道的会话/记忆/清单只读视图由 MiniBot 在创建后注入。
    """
    validate_channels(names)
    channels: dict[str, Channel] = {}
    for name in names:
        if name == "cli":
            channels[name] = CliChannel(bus)
        elif name == "web":
            channels[name] = WebChannel(bus, port=web_port)
        else:
            channels[name] = EchoChannel(bus, port=echo_port)
    return channels
