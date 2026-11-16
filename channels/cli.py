"""CLI 渠道：从终端读输入、往终端打回复（对应 nanobot 的本地 CLI 渠道）。

输入方向：stdin 读一行 -> 包装成 InboundMessage -> 丢进总线。
输出方向：入口把总线的出站消息交回 send() -> print 到 stdout。

关键细节：
- 读 stdin 必须离开事件循环线程，否则整个运行时都会被阻塞在输入上；
- 读线程用守护线程（daemon），Ctrl+C 时即使它仍卡在 input() 里，进程也能退出；
- 读线程只在"这一轮回复已经打印完"后才显示下一个提示符，避免
  ``你: 助手: ...`` 的错位；非终端输入（管道）不打印提示符。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time

from ..bus import InboundMessage, MessageBus, OutboundMessage
from .base import Channel

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
# 不带斜杠的退出词也识别：直接敲 exit/quit 是终端用户的直觉，
# 送到模型那里只会得到一句"已退出"的假回复。
_EXIT_WORDS = {"exit", "quit"}
# 快速模型通常在 0.8 秒内返回；超过这个时间才提示"正在等待"，
# 避免每条消息都闪一行无意义的等待文案。
_WAITING_HINT_DELAY = 0.8
# 状态行保持一行：参数太长会把工具名挤出屏幕，也容易撑乱正在流式输出的正文。
_MAX_ARG_CHARS = 140
_MAX_ARG_VALUE_CHARS = 48
_MAX_RESULT_PREVIEW_CHARS = 400
_MAX_RESULT_PREVIEW_LINES = 6


def _is_exit(text: str) -> bool:
    return text in _EXIT_COMMANDS or text.lower() in _EXIT_WORDS


def _clip(text: object, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 1)] + "…"


def _format_arguments(arguments: object) -> str:
    """把工具参数压成 ``key=value`` 形式，避免整段 JSON 占满终端。"""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return _clip(arguments, _MAX_ARG_CHARS)
    if not isinstance(arguments, dict) or not arguments:
        return ""
    parts = [
        f"{key}={_clip(value, _MAX_ARG_VALUE_CHARS)}"
        for key, value in list(arguments.items())[:5]
    ]
    if len(arguments) > 5:
        parts.append("…")
    return _clip(", ".join(parts), _MAX_ARG_CHARS)


def _format_duration(duration_ms: object) -> str:
    try:
        value = float(duration_ms)
    except (TypeError, ValueError):
        return ""
    if value < 1000:
        return f"{value:.0f}ms"
    return f"{value / 1000:.1f}s"


def _format_size(result: str) -> str:
    chars = len(result)
    if chars < 1000:
        return f"{chars} 字符"
    return f"{chars / 1000:.1f}k 字符"


def _format_result_preview(result: str) -> str:
    """失败时给出可定位的结果片段；成功时不刷屏，只显示体积。"""
    lines = result.strip().splitlines()[:_MAX_RESULT_PREVIEW_LINES]
    preview = "\n".join(f"    {line}" for line in lines)
    if len(preview) > _MAX_RESULT_PREVIEW_CHARS:
        preview = preview[:_MAX_RESULT_PREVIEW_CHARS] + "…"
    return preview


def _format_usage(usage: dict | None) -> str:
    if not usage:
        return ""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or 0)
    if not (prompt or completion or total):
        return ""
    return f"输入 {prompt} / 输出 {completion} / 合计 {total} tokens"


class CliChannel(Channel):
    name = "cli"

    def __init__(self, bus: MessageBus, session_key: str | None = None) -> None:
        super().__init__(bus)
        self.sender = "local"  # 本地终端只有一个"用户"
        self.session_key = session_key  # CLI 的 -s；None 时按 cli:local
        # 入口（__main__）await 这个事件就知道用户何时退出
        self.closed = asyncio.Event()
        self._task: asyncio.Task | None = None
        # 每条入站消息恰好对应一条出站回复：读循环靠它等"这一轮答完"
        self._reply_ready = asyncio.Event()
        # 流式状态：只在第一段增量时打印一次 "助手: "，最终消息负责收尾换行。
        self._stream_active = False
        self._stream_last_char = ""
        # 本轮起点：最终回复时给出总耗时（只在真正有过输入时才有值）。
        self._turn_started: float | None = None

        # 读线程与事件循环之间只通过线程安全队列通信；
        # "该读下一行了" 用 threading.Event 控制，保证提示符出现在回复之后。
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._want_line = threading.Event()
        self._stop_reader = threading.Event()
        self._reader: threading.Thread | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_running_loop()
        self._reader = threading.Thread(
            target=self._stdin_reader,
            args=(loop,),
            name="mini-agent-cli-stdin",
            daemon=True,
        )
        self._reader.start()
        self._task = asyncio.create_task(self._read_loop(), name="mini-agent-cli-read")

    def _stdin_reader(self, loop: asyncio.AbstractEventLoop) -> None:
        """守护线程：按事件循环的节奏读一行，投递后继续等待下一次请求。"""
        show_prompt = sys.stdin.isatty()
        while not self._stop_reader.is_set():
            if not self._want_line.wait(timeout=0.2):
                continue
            self._want_line.clear()
            if self._stop_reader.is_set():
                return
            try:
                line = input("你: " if show_prompt else "")
            except (EOFError, KeyboardInterrupt):
                line = None
            if self._stop_reader.is_set():
                return
            try:
                loop.call_soon_threadsafe(self._lines.put_nowait, line)
            except RuntimeError:
                return  # 事件循环已关闭，安静退出
            if line is not None and _is_exit(line.strip()):
                return

    async def _read_loop(self) -> None:
        """后台任务：取一行 -> 发布入站消息 -> 等这一轮回复打印完。"""
        while not self.closed.is_set():
            self._want_line.set()
            line = await self._lines.get()
            if line is None:  # EOF / Ctrl+C
                self.closed.set()
                return
            text = line.strip()
            if not text:
                continue
            if _is_exit(text):
                self.closed.set()
                return

            self._reply_ready.clear()
            self._turn_started = time.monotonic()
            hint_task = asyncio.create_task(self._show_waiting_hint())
            try:
                await self.bus.publish_inbound(
                    InboundMessage(
                        channel=self.name,
                        sender=self.sender,
                        text=text,
                        session_key_override=self.session_key,
                    )
                )
                # 等 send() 把这一轮的回复打印完，再回头显示下一个 "你: "。
                await self._reply_ready.wait()
            finally:
                hint_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await hint_task

    async def _show_waiting_hint(self) -> None:
        await asyncio.sleep(_WAITING_HINT_DELAY)
        # 提示只是"还没开始输出"的兜底：首个增量或整段回复已经到了就不要再插一行。
        if self._stream_active or self._reply_ready.is_set():
            return
        print("（正在等待模型回复…）", flush=True)

    async def send(self, msg: OutboundMessage) -> None:
        # 工具事件不进入回复正文：先收掉正在流式的行，再单独打印一行状态。
        if msg.event:
            self._end_stream_line()
            kind = msg.event.get("type")
            if kind == "tool_start":
                self._print_tool_start(msg.event)
            elif kind == "tool_end":
                self._print_tool_end(msg.event)
            elif kind == "stopped":
                print("（已停止生成）", flush=True)
                self._reply_ready.set()
            return

        # 流式增量：第一段打印一次前缀，后续直接追加，不换行。
        if msg.partial:
            if not msg.text:
                return
            if not self._stream_active:
                print("助手: ", end="", flush=True)
                self._stream_active = True
            print(msg.text, end="", flush=True)
            self._stream_last_char = msg.text[-1]
            return

        # 最终消息：流式期间正文已经打印过，这里只负责收尾换行；
        # 非流式（或端点回退）才把完整文本打印出来。
        if self._stream_active:
            if self._stream_last_char != "\n":
                print(flush=True)
            self._stream_active = False
            self._stream_last_char = ""
            if not msg.streamed:
                print(f"助手: {msg.text}", flush=True)
        else:
            # flush：stdout 被重定向时是块缓冲，不刷新会让回复排到下一个提示符之后
            print(f"助手: {msg.text}", flush=True)
        if msg.error:
            print("（本轮失败，可重新发送同一条消息重试）", flush=True)
        if msg.side_effects:
            print(
                "（注意：本轮已执行副作用工具 "
                + "、".join(msg.side_effects)
                + "，重试可能重复执行）",
                flush=True,
            )
        stats = self._format_stats(msg.usage)
        if stats:
            print(stats, flush=True)
        self._turn_started = None
        self._reply_ready.set()  # 放行读循环：现在可以读取下一条输入了

    @staticmethod
    def _print_tool_start(event: dict) -> None:
        name = event.get("name") or "?"
        arguments = _format_arguments(event.get("arguments"))
        suffix = "（需确认）" if event.get("side_effect") else ""
        call = f"{name}({arguments})" if arguments else str(name)
        print(f"  [工具] {call}{suffix}", flush=True)

    @staticmethod
    def _print_tool_end(event: dict) -> None:
        name = event.get("name") or "?"
        result = str(event.get("result") or "")
        if event.get("denied"):
            print(f"  [工具] {name} 未获批准", flush=True)
            return
        state = "完成" if event.get("ok") else "失败"
        details = [
            text
            for text in (
                _format_duration(event.get("duration_ms")),
                _format_size(result) if result else "",
            )
            if text
        ]
        line = f"  [工具] {name} {state}"
        if details:
            line += " · " + " · ".join(details)
        print(line, flush=True)
        if not event.get("ok") and result:
            print(_format_result_preview(result), flush=True)

    def _format_stats(self, usage: dict | None) -> str:
        """最终回复后的一行统计：耗时 + token 用量，两者都有才值得占一行。"""
        parts: list[str] = []
        if self._turn_started is not None:
            elapsed = time.monotonic() - self._turn_started
            # 本地命令通常毫秒级返回，0.0s 只是噪声，不值得占一行。
            if elapsed >= 0.1:
                parts.append(f"{elapsed:.1f}s")
        tokens = _format_usage(usage)
        if tokens:
            parts.append(tokens)
        if not parts:
            return ""
        return "（" + " · ".join(parts) + "）"

    def _end_stream_line(self) -> None:
        """工具事件插队前先结束未闭合的流式行，避免状态文字粘进回复正文。"""
        if not self._stream_active:
            return
        if self._stream_last_char != "\n":
            print(flush=True)
        self._stream_active = False
        self._stream_last_char = ""

    async def request_approval(
        self,
        *,
        session_key: str,
        sender: str,
        tool_name: str,
        arguments: dict,
    ) -> bool:
        """终端里逐次询问副作用工具；非交互输入直接拒绝。"""
        if not sys.stdin.isatty():
            print(
                f"（工具 {tool_name} 可能修改文件或执行命令，"
                "非交互模式无法确认，已拒绝）",
                flush=True,
            )
            return False
        self._end_stream_line()
        preview = json.dumps(arguments, ensure_ascii=False)
        if len(preview) > 1000:
            preview = preview[:1000] + "…"
        prompt = f"工具 {tool_name} 请求执行，参数：{preview}\n允许执行？[y/N] "
        try:
            answer = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes", "是"}

    async def stop(self) -> None:
        self._stop_reader.set()
        self._want_line.set()  # 唤醒可能在等待"下一行"的读线程
        if self._task is not None:
            self._task.cancel()
            # 取消后台读任务；若它恰好刚结束，CancelledError 也要吞掉
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
