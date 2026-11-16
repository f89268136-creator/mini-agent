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
import contextlib
import logging

from ..bus import InboundMessage, MessageBus, OutboundMessage
from .base import Channel

logger = logging.getLogger(__name__)

# 单行消息上限：防止一个客户端用超长行占满内存或拖垮解析
MAX_LINE_BYTES = 64 * 1024
# 并发连接上限：演示渠道没有鉴权，必须限制资源占用
MAX_CONNECTIONS = 32


class EchoChannel(Channel):
    name = "echo"

    def __init__(self, bus: MessageBus, port: int = 8766) -> None:
        super().__init__(bus)
        self.port = port
        self._server: asyncio.Server | None = None
        # sender -> 写端：回复时按 OutboundMessage.to（即 sender）找回连接
        self._writers: dict[str, asyncio.StreamWriter] = {}

    async def start(self) -> None:
        # 每来一条 TCP 连接就调用一次 _on_connect（由 asyncio 起协程）；
        # limit 同时约束 StreamReader 的内部缓冲，超长行不会无限占用内存。
        self._server = await asyncio.start_server(
            self._on_connect, "127.0.0.1", self.port, limit=MAX_LINE_BYTES
        )

    async def _on_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """单条连接的处理循环：读到的每一行都发布进总线。"""
        if len(self._writers) >= MAX_CONNECTIONS:
            logger.warning("echo 连接数已达上限（%d），拒绝新连接", MAX_CONNECTIONS)
            await self._close_writer(writer, "服务器忙，请稍后重试\n")
            return

        peer = writer.get_extra_info("peername") or ("?", 0)
        peer_name = f"{peer[0]}:{peer[1]}"  # 用 ip:port 区分不同客户端
        sender = f"peer:{peer_name}"
        self._writers[sender] = writer
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    # 单行超过 MAX_LINE_BYTES：StreamReader 无法继续可靠解析，断开连接
                    logger.warning("echo 客户端 %s 发送了超长消息，连接已关闭", peer_name)
                    await self._close_writer(writer, "消息过长，连接已关闭\n")
                    break
                if not line:
                    break  # 对端断开连接
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue  # 空行忽略
                await self.bus.publish_inbound(
                    InboundMessage(channel=self.name, sender=sender, text=text)
                )
        except (ConnectionError, OSError):
            logger.debug("echo 客户端 %s 连接异常断开", peer_name)
        finally:
            # 无论正常断开还是异常，都要把写端从表里摘掉
            self._writers.pop(sender, None)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def send(self, msg: OutboundMessage) -> None:
        if msg.partial:
            # 行协议一条消息一行；流式增量会把一句话拆成多行，等最终完整回复再发。
            return
        writer = self._writers.get(msg.to)
        if writer is None or writer.is_closing():
            logger.debug("echo 对端已断开，丢弃回复: to=%s", msg.to)
            return
        encoded = msg.text.encode("utf-8")
        if len(encoded) > MAX_LINE_BYTES - 16:
            # 回复也必须保持"一行"，否则客户端按行读取会拆成多条消息；
            # 按字节截断后再按 UTF-8 解码，避免切坏多字节字符。
            encoded = (
                encoded[: MAX_LINE_BYTES - 32].decode("utf-8", errors="ignore").encode("utf-8")
                + "…（已截断）".encode("utf-8")
            )
        try:
            writer.write(encoded + b"\n")  # 一行一条回复
            await writer.drain()  # 等发送缓冲排空，避免无界积压
        except (ConnectionError, OSError):
            self._writers.pop(msg.to, None)
            logger.debug("echo 对端已断开，回复发送失败: to=%s", msg.to)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        writers = list(self._writers.values())
        self._writers.clear()
        for writer in writers:
            writer.close()
        for writer in writers:
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter, notice: str) -> None:
        """发一条告示后关闭连接；对端已消失时静默忽略。"""
        try:
            writer.write(notice.encode("utf-8"))
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
