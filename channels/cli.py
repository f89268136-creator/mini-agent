"""CLI 渠道：从终端读输入、往终端打回复（对应 nanobot 的本地 CLI 渠道）。

输入方向：stdin 读一行 -> 包装成 InboundMessage -> 丢进总线。
输出方向：入口把总线的出站消息交回 send() -> print 到 stdout。

关键细节：input() 是阻塞函数，必须用 asyncio.to_thread 丢到线程里执行，
否则整个事件循环（别的渠道、核心处理）都会被它卡住。
"""

from __future__ import annotations

import asyncio
import contextlib

from ..bus import InboundMessage, MessageBus, OutboundMessage
from .base import Channel


class CliChannel(Channel):
    name = "cli"

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(bus)
        self.sender = "local"  # 本地终端只有一个"用户"
        # 入口（__main__）await 这个事件就知道用户何时退出
        self.closed = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        # 读 stdin 会一直挂起等待，不能挡住 start() 的调用方，所以放后台任务
        self._task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """后台任务：反复读 stdin 并发布入站消息，直到退出或输入流结束。"""
        while not self.closed.is_set():
            try:
                # 阻塞式 input 放线程池；提示符 "你: " 由 input 自己打印
                text = await asyncio.to_thread(input, "你: ")
            except (EOFError, KeyboardInterrupt):
                self.closed.set()  # Ctrl+C 或 stdin 关闭 = 退出
                return
            text = text.strip()
            if not text:
                continue  # 空行忽略
            if text == "/exit":
                self.closed.set()
                return
            await self.bus.publish_inbound(
                InboundMessage(channel=self.name, sender=self.sender, text=text)
            )

    async def send(self, msg: OutboundMessage) -> None:
        print(f"助手: {msg.text}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            # 取消后台读任务；若它恰好刚结束，CancelledError 也要吞掉
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
