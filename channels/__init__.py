"""渠道包：渠道抽象 + 两个实现 + 最小工厂。

对应 nanobot 的 channels/manager.py 的最小版：nanobot 用 pkgutil 扫描
目录自动发现所有渠道类，再按配置启用；这里简化成一张"名字 -> 类"的
显式映射表，够用、不魔法，方便教学时一眼看穿装配过程。
"""

from __future__ import annotations

from ..bus import MessageBus
from .base import Channel
from .cli import CliChannel
from .echo import EchoChannel

__all__ = ["Channel", "CliChannel", "EchoChannel", "build_channels"]


def build_channels(
    names: list[str], bus: MessageBus, *, echo_port: int = 8766
) -> dict[str, Channel]:
    """按配置里的名字列表创建渠道实例，返回 {name: channel}。

    未知名字给出中文提示并跳过——配置写错了不至于让程序起不来。
    """
    channels: dict[str, Channel] = {}
    for name in names:
        if name == "cli":
            channels[name] = CliChannel(bus)
        elif name == "echo":
            channels[name] = EchoChannel(bus, port=echo_port)
        else:
            print(f"[mini_agent] 未知渠道名：{name}，已跳过（可选：cli, echo）")
    return channels
