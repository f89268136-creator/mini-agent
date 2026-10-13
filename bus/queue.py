"""部件 6：消息总线（对应 nanobot 的 bus/queue.py + bus/events.py 的最小版）。

为什么要总线？渠道（CLI、Echo、以后的 Telegram……）只管"收发消息"，
Agent 核心只管"处理消息"，两者之间垫两个队列：
  渠道 --publish_inbound--> [入站队列] --consume_inbound--> 核心
  核心 --publish_outbound--> [出站队列] --consume_outbound--> 渠道
双方互不认识，新增一个渠道不用改核心——这就是"解耦"。

nanobot 生产版的 MessageBus 还有本地订阅者、事件通道（publish_event）、
运行时事件等；本教学版刻意只保留最小子集：两个 asyncio.Queue。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class InboundMessage:
    """从渠道收到的消息（外部世界 -> 核心）。"""

    channel: str  # 渠道名，如 "cli" / "echo"
    sender: str  # 渠道内的发送者标识
    text: str  # 消息文本

    @property
    def session_key(self) -> str:
        """会话分片键：channel + sender。

        nanobot 用 session key 把同会话消息路由到同一个处理分片，
        保证"同一个人"的连续消息被串行处理、共享同一份历史记忆，
        不同的人/渠道之间互不干扰。
        """
        return f"{self.channel}:{self.sender}"


@dataclass
class OutboundMessage:
    """要发回渠道的消息（核心 -> 外部世界）。"""

    channel: str  # 发给哪个渠道（决定路由）
    to: str  # 渠道内的目标标识（通常是入站消息的 sender）
    text: str  # 回复文本


class MessageBus:
    """两个 asyncio.Queue 组成的极简消息总线。"""

    def __init__(self) -> None:
        # 队列在"谁先收/谁先发"上都是线程内异步安全的，直接当管道用
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """渠道调用：把收到的消息放进入站队列（put 不会被阻塞卡住核心）。"""
        await self._inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """核心调用：取一条入站消息；队列空时挂起等待。"""
        return await self._inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """核心调用：把回复放进出站队列。"""
        await self._outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """渠道调用：取一条出站消息去发送。"""
        return await self._outbound.get()

    # 下面两个 size() 仅供教学观测（例如验证消息确实入队了），nanobot 本身没有。
    def inbound_size(self) -> int:
        return self._inbound.qsize()

    def outbound_size(self) -> int:
        return self._outbound.qsize()
