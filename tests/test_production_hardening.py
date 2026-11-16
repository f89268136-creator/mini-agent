"""生产加固回归测试：数据隔离、历史一致性、超时与停机语义。

全部离线运行：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python.exe \
        -m unittest mini_agent.tests.test_production_hardening -v
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest

from mini_agent import sdk
from mini_agent.agent import Agent, AgentLoop
from mini_agent.agent.tools.registry import (
    REGISTRY,
    configure_tool_timeout,
    register_tool,
)
from mini_agent.bus import MessageBus, OutboundMessage
from mini_agent.config import ConfigError, MiniConfig
from mini_agent.memory.store import MemoryStore
from mini_agent.providers.base import Message, ToolCall
from mini_agent.session import SessionManager
from mini_agent.session.todo import TodoStore
from mini_agent.utils.naming import safe_key


class KeyIsolationTests(unittest.TestCase):
    def test_safe_key_separates_sanitized_collisions(self):
        self.assertNotEqual(safe_key("a:b"), safe_key("a_b"))
        self.assertNotEqual(safe_key("ClI:Local"), safe_key("cli:local"))

    def test_sessions_with_colliding_sanitized_names_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(tmp, "SYS")
            first = manager.get_or_create("a:b")
            first.add_message(Message(role="user", content="冒号租户"))
            asyncio.run(manager.save(first))

            second = manager.get_or_create("a_b")
            second.add_message(Message(role="user", content="下划线租户"))
            asyncio.run(manager.save(second))

            reloaded = SessionManager(tmp, "SYS")
            colon = [m.content for m in reloaded.get_or_create("a:b").messages]
            underscore = [m.content for m in reloaded.get_or_create("a_b").messages]
            self.assertIn("冒号租户", colon)
            self.assertNotIn("下划线租户", colon)
            self.assertIn("下划线租户", underscore)
            self.assertNotIn("冒号租户", underscore)
            files = [name for name in os.listdir(tmp) if name.endswith(".jsonl")]
            self.assertEqual(len(files), 2)

    def test_memory_store_isolates_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.add("a:b", "冒号记忆")
            store.add("a_b", "下划线记忆")
            self.assertEqual([e.text for e in store.entries("a:b")], ["冒号记忆"])
            self.assertEqual([e.text for e in store.entries("a_b")], ["下划线记忆"])
            self.assertNotEqual(store._path("a:b"), store._path("a_b"))

    def test_todo_store_isolates_colliding_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TodoStore(tmp)
            store.write("a:b", [{"content": "冒号任务"}])
            store.write("a_b", [{"content": "下划线任务"}])
            self.assertEqual([i.content for i in store.items("a:b")], ["冒号任务"])
            self.assertEqual([i.content for i in store.items("a_b")], ["下划线任务"])


class _StubProvider:
    """Agent 构造只需要一个 provider 占位；测试里不真正调用模型。"""


class LightweightAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncation_drops_orphan_tool_messages(self):
        agent = Agent(_StubProvider(), history_limit=2)
        call = ToolCall(id="call-1", name="now", arguments="{}")
        agent.history.extend(
            [
                Message(role="assistant", tool_calls=[call]),
                Message(role="tool", content="工具结果", tool_call_id="call-1"),
            ]
        )
        captured: dict[str, list[Message]] = {}

        async def fake_run(messages, **kwargs):
            captured["messages"] = list(messages)
            return "ok"

        agent.runner.run = fake_run
        self.assertEqual(await agent.chat("新问题"), "ok")
        self.assertEqual(
            [m.role for m in captured["messages"]], ["system", "user"]
        )


class _BrokenRunner:
    async def run(self, messages, on_delta=None, on_event=None, usage=None):
        raise RuntimeError("provider down")


class _SlowRunner:
    async def run(self, messages, on_delta=None, on_event=None, usage=None):
        await asyncio.sleep(0.5)
        return "迟到的回答"


class _CountingRunner:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def run(self, messages, on_delta=None, on_event=None, usage=None):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.05)
        finally:
            self.active -= 1
        return "ok"


def _make_loop(tmp: str, runner, **kwargs) -> AgentLoop:
    return AgentLoop(
        MessageBus(),
        runner,
        SessionManager(tmp, "SYS", history_limit=50),
        session_idle_timeout=0,
        **kwargs,
    )


class LoopFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_turn_rolls_back_user_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = _make_loop(tmp, _BrokenRunner())
            with self.assertRaises(RuntimeError):
                await loop.handle_text(channel="sdk", sender="u", text="会失败的一句")
            session = loop.sessions.get_or_create("sdk:u")
            self.assertEqual([m.role for m in session.messages], ["system"])
            self.assertFalse(os.path.exists(loop.sessions._path("sdk:u")))

    async def test_turn_timeout_rolls_back_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = _make_loop(tmp, _SlowRunner(), turn_timeout=0.05)
            with self.assertRaises(TimeoutError) as caught:
                await loop.handle_text(channel="sdk", sender="u", text="慢请求")
            self.assertIn("本轮处理超过", str(caught.exception))
            session = loop.sessions.get_or_create("sdk:u")
            self.assertEqual([m.role for m in session.messages], ["system"])

    async def test_max_concurrent_turns_limits_parallel_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _CountingRunner()
            loop = _make_loop(tmp, runner, max_concurrent_turns=1)
            await asyncio.gather(
                loop.handle_text(channel="sdk", sender="a", text="甲"),
                loop.handle_text(channel="sdk", sender="b", text="乙"),
            )
            self.assertEqual(runner.peak, 1)


class ToolDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        configure_tool_timeout(0)

    async def asyncTearDown(self):
        configure_tool_timeout(0)
        REGISTRY.pop("test_slow_tool", None)

    async def test_sync_tool_deadline(self):
        def slow_tool() -> str:
            time.sleep(0.15)
            return "done"

        register_tool(slow_tool, name="test_slow_tool", description="测试用慢工具")
        configure_tool_timeout(0.02)
        ok, result = await REGISTRY["test_slow_tool"].arun_with_status({})
        self.assertFalse(ok)
        self.assertIn("工具执行超时", result)


class _SlowChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.started = asyncio.Event()

    async def send(self, message: OutboundMessage) -> None:
        self.started.set()
        await asyncio.sleep(0.1)
        self.sent.append(message.text)


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_waits_for_inflight_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            channel = _SlowChannel()
            loop = AgentLoop(
                MessageBus(),
                _BrokenRunner(),
                SessionManager(tmp, "SYS"),
                channels={"fake": channel},
                session_idle_timeout=0,
            )
            await loop.start()
            await loop.bus.publish_outbound(
                OutboundMessage(channel="fake", to="u", text="最后一条")
            )
            await asyncio.wait_for(channel.started.wait(), timeout=1)
            self.assertEqual(loop.stats()["outbound_busy"], 1)

            await loop.stop(graceful_timeout=1.0)
            self.assertEqual(channel.sent, ["最后一条"])


class ConfigAndGlobalsTests(unittest.TestCase):
    def test_new_limits_validate(self):
        config = MiniConfig(turn_timeout=0, tool_timeout=0, max_concurrent_turns=0)
        self.assertEqual(config.tool_timeout, 0.0)
        with self.assertRaises(ConfigError):
            MiniConfig(turn_timeout=-1)
        with self.assertRaises(ConfigError):
            MiniConfig(tool_timeout=-1)

    def test_conflicting_process_globals_warn(self):
        original = sdk._configured_fingerprint
        sdk._configured_fingerprint = None
        try:
            with self.assertLogs("mini_agent.sdk", level="WARNING"):
                sdk._check_process_globals(MiniConfig(workspace="a"))
                sdk._check_process_globals(MiniConfig(workspace="b"))
        finally:
            sdk._configured_fingerprint = original


if __name__ == "__main__":
    unittest.main()
