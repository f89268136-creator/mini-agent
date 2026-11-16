"""故障注入与安全边界回归测试：重试、工具异常、审批与引用。

全部离线运行：
    .venv/Scripts/python.exe -m unittest mini_agent.tests.test_resilience -v
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mini_agent.agent.loop import AgentLoop
from mini_agent.agent.request import RequestContext, request_scope
from mini_agent.agent.runner import AgentRunner
from mini_agent.agent.tools.knowledge import (
    citation_payload,
    citation_scope,
    configure_knowledge_base,
    knowledge_search,
    reset_knowledge_base,
)
from mini_agent.agent.tools.policy import configure_policy
from mini_agent.agent.tools.registry import Tool
from mini_agent.bus import MessageBus
from mini_agent.providers.base import LLMProvider, LLMResponse, Message, Usage
from mini_agent.providers.embeddings import HashingEmbedder
from mini_agent.providers.openai_compat import (
    OpenAICompatProvider,
    RetryingProvider,
    _is_transient_error,
    _retry_after_seconds,
)
from mini_agent.session import SessionManager


class _TimeoutError(Exception):
    """类名包含 timeout，走非 openai SDK 异常的兜底分类。"""


class _FakeProvider(LLMProvider):
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def chat(self, messages, tools=None, on_delta=None):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            if on_delta is not None and getattr(outcome, "emit_first", False):
                await on_delta("部分输出")
            raise outcome
        if on_delta is not None and outcome.content:
            await on_delta(outcome.content)
        return outcome


def _ok(text: str = "完成") -> LLMResponse:
    return LLMResponse(content=text, usage=Usage(1, 2, 3))


class RetryClassificationTests(unittest.TestCase):
    def test_transient_errors_are_recognised(self) -> None:
        self.assertTrue(_is_transient_error(_TimeoutError("timeout")))
        self.assertTrue(_is_transient_error(ConnectionError("connection reset")))
        self.assertTrue(_is_transient_error(RuntimeError("HTTP 429 too many requests")))
        self.assertFalse(_is_transient_error(ValueError("invalid api key")))

    def test_retry_after_is_clamped(self) -> None:
        exc = SimpleNamespace(
            response=SimpleNamespace(headers={"retry-after": "120"})
        )
        self.assertEqual(_retry_after_seconds(exc), 30.0)
        bad = SimpleNamespace(
            response=SimpleNamespace(headers={"retry-after": "not-a-number"})
        )
        self.assertIsNone(_retry_after_seconds(bad))


class RetryingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_failure_is_retried(self) -> None:
        inner = _FakeProvider([_TimeoutError("timeout"), _ok("第二次成功")])
        provider = RetryingProvider(inner, max_attempts=3, base_delay=0)
        response = await provider.chat([Message(role="user", content="hi")])
        self.assertEqual(response.content, "第二次成功")
        self.assertEqual(inner.calls, 2)

    async def test_permanent_failure_is_not_retried(self) -> None:
        inner = _FakeProvider([ValueError("invalid api key")])
        provider = RetryingProvider(inner, max_attempts=3, base_delay=0)
        with self.assertRaises(ValueError):
            await provider.chat([Message(role="user", content="hi")])
        self.assertEqual(inner.calls, 1)

    async def test_failure_after_partial_output_is_not_retried(self) -> None:
        exc = _TimeoutError("timeout")
        exc.emit_first = True  # type: ignore[attr-defined]
        inner = _FakeProvider([exc, _ok("不该出现")])
        provider = RetryingProvider(inner, max_attempts=3, base_delay=0)
        deltas: list[str] = []

        async def on_delta(text: str) -> None:
            deltas.append(text)

        with self.assertRaises(_TimeoutError):
            await provider.chat(
                [Message(role="user", content="hi")], on_delta=on_delta
            )
        self.assertEqual(inner.calls, 1)
        self.assertEqual(deltas, ["部分输出"])

    async def test_gateway_error_finish_reason_is_retried(self) -> None:
        inner = _FakeProvider(
            [
                LLMResponse(content=None, finish_reason="error"),
                _ok("恢复"),
            ]
        )
        provider = RetryingProvider(inner, max_attempts=3, base_delay=0)
        response = await provider.chat([Message(role="user", content="hi")])
        self.assertEqual(response.content, "恢复")
        self.assertEqual(inner.calls, 2)

    async def test_usage_is_preserved_on_success(self) -> None:
        inner = _FakeProvider([_ok("x")])
        provider = RetryingProvider(inner, max_attempts=1, base_delay=0)
        response = await provider.chat([Message(role="user", content="hi")])
        self.assertEqual(response.usage.total_tokens, 3)


class ProviderParsingTests(unittest.TestCase):
    def test_empty_choices_raise(self) -> None:
        raw = SimpleNamespace(choices=[])
        with self.assertRaises(RuntimeError):
            OpenAICompatProvider._parse_response(raw)

    def test_missing_usage_stays_none(self) -> None:
        message = SimpleNamespace(content="答案", tool_calls=None)
        raw = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )
        response = OpenAICompatProvider._parse_response(raw)
        self.assertEqual(response.content, "答案")
        self.assertIsNone(response.usage)


class ToolFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_exception_is_converted_to_text(self) -> None:
        def boom() -> str:
            raise RuntimeError("炸了")

        tool = Tool(
            name="boom",
            description="总是失败",
            func=boom,
            parameters={},
        )
        ok, rendered = await tool.arun_with_status({})
        self.assertFalse(ok)
        self.assertIn("RuntimeError", rendered)
        self.assertIn("炸了", rendered)


class ApprovalGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        configure_policy(tool_approval="ask")

    async def asyncTearDown(self) -> None:
        configure_policy(tool_approval="ask")

    @staticmethod
    def _tool() -> Tool:
        return Tool(
            name="write_file",
            description="写文件",
            func=lambda **_: "ok",
            parameters={},
            requires_approval=True,
        )

    async def test_no_channel_capability_denies(self) -> None:
        self.assertFalse(await AgentRunner._approved(self._tool(), {"path": "a"}))

    async def test_callback_approves_and_denies(self) -> None:
        async def approve(name: str, arguments: dict) -> bool:
            return arguments.get("path") == "ok.txt"

        context = RequestContext(
            channel="test",
            sender="tester",
            session_key="test:1",
            approval=approve,
        )
        with request_scope(context):
            self.assertTrue(
                await AgentRunner._approved(self._tool(), {"path": "ok.txt"})
            )
            self.assertFalse(
                await AgentRunner._approved(self._tool(), {"path": "no.txt"})
            )

    async def test_callback_failure_denies(self) -> None:
        async def broken(name: str, arguments: dict) -> bool:
            raise RuntimeError("channel down")

        context = RequestContext(
            channel="test",
            sender="tester",
            session_key="test:1",
            approval=broken,
        )
        with request_scope(context):
            self.assertFalse(
                await AgentRunner._approved(self._tool(), {"path": "a"})
            )

    async def test_mode_off_skips_approval(self) -> None:
        configure_policy(tool_approval="off")
        self.assertTrue(await AgentRunner._approved(self._tool(), {"path": "a"}))

    async def test_tool_without_flag_skips_approval(self) -> None:
        tool = Tool(
            name="read_file",
            description="只读",
            func=lambda **_: "ok",
            parameters={},
        )
        self.assertTrue(await AgentRunner._approved(tool, {"path": "a"}))


class _SideEffectRunner:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    async def run(self, messages, on_delta=None, on_event=None, usage=None):
        if on_event is not None:
            await on_event(
                {
                    "type": "tool_start",
                    "id": "c1",
                    "name": "write_file",
                    "arguments": {"path": "a.txt"},
                    "side_effect": True,
                }
            )
            await on_event(
                {
                    "type": "tool_end",
                    "id": "c1",
                    "name": "write_file",
                    "ok": True,
                    "side_effect": True,
                    "result": "ok",
                }
            )
        if self.fail:
            raise RuntimeError("模型连接中断")
        return "写好了"


class FailedSideEffectTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_turn_reports_executed_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(
                MessageBus(),
                _SideEffectRunner(fail=True),
                SessionManager(tmp, "SYS", history_limit=50),
                session_idle_timeout=0,
            )
            with self.assertRaises(RuntimeError):
                await loop.handle_text(channel="sdk", sender="u", text="写文件")
            self.assertEqual(
                loop.take_failed_side_effects("sdk:u"), ["write_file"]
            )
            self.assertIsNone(loop.take_failed_side_effects("sdk:u"))

    async def test_successful_turn_clears_side_effect_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(
                MessageBus(),
                _SideEffectRunner(fail=False),
                SessionManager(tmp, "SYS", history_limit=50),
                session_idle_timeout=0,
            )
            await loop.handle_text(channel="sdk", sender="u", text="写文件")
            self.assertIsNone(loop.take_failed_side_effects("sdk:u"))


class CitationScopeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        configure_knowledge_base(
            HashingEmbedder(),
            chunk_size=200,
            overlap=20,
            index_dir=Path(self.tmp.name) / "kb",
        )
        self.addCleanup(reset_knowledge_base)
        from mini_agent.agent.tools.knowledge import get_knowledge_base

        kb = get_knowledge_base()
        await kb.ingest(
            "# 部署\n\n上线前先运行测试，再打标签。\n\n# 回滚\n\n保留上一版本镜像。",
            source="ops.md",
            replace_source=True,
        )

    async def test_search_hits_are_collected_per_turn(self) -> None:
        with citation_scope() as bucket:
            await asyncio.to_thread(knowledge_search, "上线前先运行测试", 2)
            payload = citation_payload(bucket)
        self.assertTrue(payload)
        first = payload[0]
        self.assertEqual(first["source"], "ops.md")
        self.assertGreaterEqual(first["start_line"], 1)
        self.assertGreaterEqual(first["end_line"], first["start_line"])
        self.assertTrue(first["chunk_id"])
        self.assertIn("snippet", first)

    async def test_empty_scope_yields_no_citations(self) -> None:
        hits = []
        with citation_scope() as bucket:
            hits = list(bucket)
        self.assertEqual(hits, [])
        self.assertEqual(citation_payload(hits), [])


if __name__ == "__main__":
    unittest.main()
