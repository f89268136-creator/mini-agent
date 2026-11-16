"""CLI 终端输出格式的回归测试（不启动读线程，只测 send 的渲染）。

    .venv/Scripts/python.exe -m unittest mini_agent.tests.test_cli_rendering -v
"""

from __future__ import annotations

import contextlib
import io
import unittest

from mini_agent.bus import MessageBus, OutboundMessage
from mini_agent.channels.cli import CliChannel


class CliRenderingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.channel = CliChannel(MessageBus())

    async def _send(self, msg: OutboundMessage) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            await self.channel.send(msg)
        return buffer.getvalue()

    async def test_tool_events_use_name_field_and_compact_arguments(self) -> None:
        start = await self._send(
            OutboundMessage(
                channel="cli",
                to="local",
                text="",
                event={
                    "type": "tool_start",
                    "name": "read_file",
                    "arguments": {"path": "README.md", "max_lines": 40},
                    "side_effect": False,
                },
            )
        )
        self.assertIn("[工具] read_file(path=README.md, max_lines=40)", start)
        self.assertNotIn("?", start)

    async def test_tool_end_shows_duration_and_size_on_success(self) -> None:
        output = await self._send(
            OutboundMessage(
                channel="cli",
                to="local",
                text="",
                event={
                    "type": "tool_end",
                    "name": "grep",
                    "ok": True,
                    "duration_ms": 12.4,
                    "result": "a.py:1: x\n" * 10,
                },
            )
        )
        self.assertIn("[工具] grep 完成 · 12ms", output)
        self.assertIn("字符", output)

    async def test_tool_failure_prints_preview_and_denied_is_explicit(self) -> None:
        failed = await self._send(
            OutboundMessage(
                channel="cli",
                to="local",
                text="",
                event={
                    "type": "tool_end",
                    "name": "write_file",
                    "ok": False,
                    "duration_ms": 3,
                    "result": "目标路径超出工作区",
                },
            )
        )
        self.assertIn("write_file 失败", failed)
        self.assertIn("目标路径超出工作区", failed)

        denied = await self._send(
            OutboundMessage(
                channel="cli",
                to="local",
                text="",
                event={
                    "type": "tool_end",
                    "name": "run_shell",
                    "ok": False,
                    "denied": True,
                    "result": "",
                },
            )
        )
        self.assertIn("run_shell 未获批准", denied)

    async def test_streamed_reply_is_not_duplicated_and_stats_are_printed(self) -> None:
        self.channel._turn_started = 0.0  # 让耗时行稳定出现
        partial = await self._send(
            OutboundMessage(channel="cli", to="local", text="你好", partial=True)
        )
        self.assertEqual(partial, "助手: 你好")
        final = await self._send(
            OutboundMessage(
                channel="cli",
                to="local",
                text="你好",
                streamed=True,
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            )
        )
        self.assertNotIn("助手: 你好", final)
        self.assertIn("输入 10 / 输出 2 / 合计 12 tokens", final)
        self.assertIn("s ·", final)

    async def test_non_streamed_reply_and_argument_string(self) -> None:
        start = await self._send(
            OutboundMessage(
                channel="cli",
                to="local",
                text="",
                event={
                    "type": "tool_start",
                    "name": "grep",
                    "arguments": '{"pattern": "TODO", "path": "."}',
                },
            )
        )
        self.assertIn("grep(pattern=TODO, path=.)", start)
        final = await self._send(OutboundMessage(channel="cli", to="local", text="完成"))
        self.assertIn("助手: 完成", final)


if __name__ == "__main__":
    unittest.main()
