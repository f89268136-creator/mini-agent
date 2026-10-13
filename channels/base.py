"""渠道抽象基类（对应 nanobot 的 channels/base.py 的最小版）。

渠道 = 消息的"进出口"：
  入方向：外部世界的输入 -> InboundMessage -> 丢进总线；
  出方向：总线的 OutboundMessage -> 翻译回外部世界发出去。
Agent 核心只认这三个方法（start/send/stop），
所以 CLI、Echo、乃至 Telegram 可以互换——这就是"面向接口编程"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..bus import MessageBus, OutboundMessage


class Channel(ABC):
    """所有渠道的公共接口。"""

    name: str = ""  # 渠道名，必须与消息里的 channel 字段一致

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus  # 手里有总线，才能收（publish_inbound）发（consume_outbound）

    @abstractmethod
    async def start(self) -> None:
        """启动渠道：开始监听外部输入（通常起后台任务）。"""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """把一条出站消息真正发出去。"""

    @abstractmethod
    async def stop(self) -> None:
        """停止渠道并释放资源（关闭连接、取消任务等）。"""
