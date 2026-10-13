"""Echo 渠道：一个最小的 TCP 服务端（演示"第二个渠道"如何接入总线）。

为什么写它？MessageBus 的价值在于解耦——同一个核心要同时服务多个渠道。
nanobot 有 Telegram / Discord / Slack 等 16 个渠道，但接口都同形：
start() 监听平台事件并 publish_inbound，send() 把回复发回平台。
这里用最朴素的 TCP 行协议代替真实平台，让你能亲手验证
"连上端口发一行 -> Agent 处理 -> 收到一行回复"的完整链路。

协议：连接 127.0.0.1:<port>，每行一条消息，UTF-8 编码。
"""

from __future__ import annotations

import asyncio

from ..bus import InboundMessage, MessageBus, OutboundMessage
from .base import Channel


class EchoChannel(Channel):
    name = "echo"

    def __init__(self, bus: MessageBus, port: int = 8766) -> None:
        super().__init__(bus)
        self.port = port
        self._server: asyncio.Server | None = None
        # sender -> 写端：回复时按 OutboundMessage.to（即 sender）找回连接
        self._writers: dict[str, asyncio.StreamWriter] = {}

    async def start(self) -> None:
        # 每来一条 TCP 连接就调用一次 _on_connect（由 asyncio 起协程）
        self._server = await asyncio.start_server(self._on_connect, "127.0.0.1", self.port)

    async def _on_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """单条连接的处理循环：读到的每一行都发布进总线。"""
        peer = writer.get_extra_info("peername") or ("?", 0)
        peer_name = f"{peer[0]}:{peer[1]}"  # 用 ip:port 区分不同客户端
        sender = f"peer:{peer_name}"
        self._writers[sender] = writer
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # 对端断开连接
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue  # 空行忽略
                await self.bus.publish_inbound(
                    InboundMessage(channel=self.name, sender=sender, text=text)
                )
        finally:
            # 无论正常断开还是异常，都要把写端从表里摘掉
            self._writers.pop(sender, None)
            writer.close()

    async def send(self, msg: OutboundMessage) -> None:
        writer = self._writers.get(msg.to)
        if writer is None:
            return  # 对端已断开：教学版静默丢弃，不做离线队列/重试
        writer.write((msg.text + "\n").encode("utf-8"))  # 一行一条回复
        await writer.drain()  # 等发送缓冲排空，避免无界积压

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for writer in list(self._writers.values()):
            writer.close()
        self._writers.clear()
