"""部件 6：消息总线（对应 nanobot 的 bus/queue.py + bus/events.py 的最小版）。

为什么要总线？渠道（CLI、Echo、以后的 Telegram……）只管"收发消息"，
Agent 核心只管"处理消息"，两者之间垫两个队列：
  渠道 --publish_inbound--> [入站队列] --consume_inbound--> 核心
  核心 --publish_outbound--> [出站队列] --consume_outbound--> 渠道
双方互不认识，新增一个渠道不用改核心——这就是"解耦"。

nanobot 的 MessageBus 还有本地订阅者、事件通道（publish_event）、
运行时事件等；本实现只保留核心子集：两个 asyncio.Queue。
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
    session_key_override: str | None = None  # 渠道显式指定会话键（CLI 的 -s）

    @property
    def session_key(self) -> str:
        """会话分片键：channel + sender，渠道可显式覆盖。

        nanobot 用 session key 把同会话消息路由到同一个处理分片，
        保证"同一个人"的连续消息被串行处理、共享同一份历史记忆，
        不同的人/渠道之间互不干扰。
        """
        return self.session_key_override or f"{self.channel}:{self.sender}"


@dataclass
class OutboundMessage:
    """要发回渠道的消息（核心 -> 外部世界）。

    partial=True 表示这是流式输出中的一段增量，渠道应追加显示而不结束当前回合；
    streamed=True 表示最终回复的正文已经通过 partial 增量发过，渠道不要重复打印。
    不支持流式的渠道（如 echo 行协议）可以忽略 partial，等待最终消息。

    ``event`` 用于传输非文本事件（工具开始/结束、停止生成等）；设置后 ``text``
    可以为空。``citations``/``usage``/``error`` 只出现在最终消息上，Web 前端
    据此渲染来源卡片、token 统计和可重试错误。
    """

    channel: str  # 发给哪个渠道（决定路由）
    to: str  # 渠道内的目标标识（通常是入站消息的 sender）
    text: str  # 回复文本
    partial: bool = False
    streamed: bool = False
    event: dict | None = None
    citations: list[dict] | None = None
    usage: dict | None = None
    error: str | None = None
    retryable: bool = False
    # 失败回合里已经执行过的副作用工具名：重试前提醒用户，避免重复写入/执行。
    side_effects: list[str] | None = None


class MessageBus:
    """两个有界 asyncio.Queue 组成的极简消息总线。

    有界队列是过载保护：渠道被攻击或核心变慢时，内存不会随消息数无限增长，
    生产端会在队列满时被反压（``await put`` 挂起），而不是把进程撑爆。
    """

    def __init__(self, max_queue_size: int = 256) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size 必须 >= 1")
        self.max_queue_size = max_queue_size
        # 队列在"谁先收/谁先发"上都是线程内异步安全的，直接当管道用
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=max_queue_size)
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=max_queue_size)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """渠道调用：放进入站队列；队列满时等待消费，形成反压。"""
        await self._inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """核心调用：取一条入站消息；队列空时挂起等待。"""
        return await self._inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """核心调用：把回复放进出站队列；队列满时等待渠道消费。"""
        await self._outbound.put(msg)

    def publish_outbound_nowait(self, msg: OutboundMessage) -> bool:
        """尽力投递（用于过载拒绝提示）：队列满时立即返回 False，不阻塞分发器。"""
        try:
            self._outbound.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            return False

    async def consume_outbound(self) -> OutboundMessage:
        """渠道调用：取一条出站消息去发送。"""
        return await self._outbound.get()

    # 队列长度用于运行观测（stats / 停机判断），不暴露底层队列对象。
    def inbound_size(self) -> int:
        return self._inbound.qsize()

    def outbound_size(self) -> int:
        return self._outbound.qsize()
