"""自动上下文压缩与 token 预算回归测试。

全部离线运行：
    .venv/Scripts/python.exe -m unittest mini_agent.tests.test_context_compaction -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mini_agent.agent.context import ContextBuilder
from mini_agent.agent.loop import AgentLoop
from mini_agent.agent.runner import AgentRunner
from mini_agent.agent.token_budget import (
    estimate_messages_tokens,
    estimate_tokens,
    estimate_tool_schemas_tokens,
)
from mini_agent.bus import MessageBus
from mini_agent.providers.base import LLMProvider, LLMResponse, Message, ToolCall
from mini_agent.session import SessionManager
from mini_agent.utils.naming import safe_key


class _SummaryProvider(LLMProvider):
    """摘要请求返回固定摘要，普通回合返回固定回答。"""

    def __init__(self, summary: str = "摘要：用户在排查项目结构。") -> None:
        self.summary = summary
        self.requests: list[list[Message]] = []

    async def chat(self, messages, tools=None, on_delta=None):
        self.requests.append(list(messages))
        if messages and "对话压缩器" in (messages[0].content or ""):
            return LLMResponse(content=self.summary)
        return LLMResponse(content="收到")


class _BrokenSummaryProvider(_SummaryProvider):
    async def chat(self, messages, tools=None, on_delta=None):
        if messages and "对话压缩器" in (messages[0].content or ""):
            self.requests.append(list(messages))
            raise RuntimeError("summarizer down")
        return await super().chat(messages, tools, on_delta)

    def summary_requests(self) -> list[list[Message]]:
        return [
            request
            for request in self.requests
            if request and "对话压缩器" in (request[0].content or "")
        ]


def _make_loop(tmp: str, provider, **kwargs) -> AgentLoop:
    prompt_max_tokens = kwargs.pop("prompt_max_tokens", 0)
    context = ContextBuilder(
        "SYS",
        max_chars=100_000,
        max_tokens=prompt_max_tokens,
        workspace=tmp,
    )
    return AgentLoop(
        MessageBus(),
        AgentRunner(provider, max_iterations=2),
        SessionManager(tmp, "SYS", history_limit=50),
        session_idle_timeout=0,
        context=context,
        prompt_max_tokens=prompt_max_tokens,
        **kwargs,
    )


async def _seed_turns(loop: AgentLoop, count: int, chars: int = 120) -> None:
    session = loop.sessions.get_or_create("sdk:u")
    session.add_message(Message(role="user", content="查一下文件"))
    session.add_message(
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="read_file", arguments='{"path":"a.py"}')],
        )
    )
    session.add_message(Message(role="tool", content="文件内容", tool_call_id="c1"))
    session.add_message(Message(role="assistant", content="已经查到了"))
    for index in range(count):
        session.add_message(
            Message(role="user", content=f"旧问题{index}：" + "内容" * chars)
        )
        session.add_message(
            Message(role="assistant", content=f"旧回答{index}：" + "答复" * chars)
        )
    await loop.sessions.save(session)


def _summary_requests(provider: _SummaryProvider) -> list[list[Message]]:
    return [
        request
        for request in provider.requests
        if request and "对话压缩器" in (request[0].content or "")
    ]


class TokenEstimateTests(unittest.TestCase):
    def test_cjk_and_ascii_estimation(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("你好世界"), 4)
        self.assertEqual(estimate_tokens("abcdefgh"), 2)

    def test_tool_arguments_count_towards_message(self) -> None:
        plain = Message(role="assistant", content="ok")
        with_tool = Message(
            role="assistant",
            content="ok",
            tool_calls=[ToolCall(id="c", name="read_file", arguments="x" * 400)],
        )
        self.assertGreater(
            estimate_messages_tokens([with_tool]),
            estimate_messages_tokens([plain]),
        )

    def test_tool_schema_tokens_estimation(self) -> None:
        self.assertEqual(estimate_tool_schemas_tokens([]), 0)
        schema = [{"type": "function", "function": {"name": "read_file"}}]
        self.assertGreater(estimate_tool_schemas_tokens(schema), 0)


class SessionSummaryPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_round_trips_and_reset_clears_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(tmp, "SYS")
            session = manager.get_or_create("sdk:u")
            session.add_message(Message(role="user", content="你好"))
            session.add_message(Message(role="assistant", content="你好呀"))
            session.summary = "用户打了招呼。"
            session.summary_messages = 6
            await manager.save(session)
            await manager.evict("sdk:u")

            reloaded = manager.get_or_create("sdk:u")
            self.assertEqual(reloaded.summary, "用户打了招呼。")
            self.assertEqual(reloaded.summary_messages, 6)
            self.assertEqual(len(reloaded.messages), 3)

            meta_path = Path(tmp) / f"{safe_key('sdk:u')}.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["summary"], "用户打了招呼。")

            reloaded.reset()
            await manager.save(reloaded)
            await manager.evict("sdk:u")
            cleared = manager.get_or_create("sdk:u")
            self.assertEqual(cleared.summary, "")
            self.assertEqual([message.role for message in cleared.messages], ["system"])


class AutoCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_compacts_old_turns_and_persists_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _SummaryProvider()
            loop = _make_loop(
                tmp,
                provider,
                # 预算需覆盖 1024 token 的输出预留，压缩后应一次成型、不再二次压缩。
                prompt_max_tokens=1_600,
                context_compact_ratio=1.0,
                context_keep_turns=2,
            )
            await _seed_turns(loop, count=8, chars=40)
            response = await loop.handle_text(
                channel="sdk", sender="u", text="当前问题"
            )
            self.assertEqual(response.text, "收到")

            requests = _summary_requests(provider)
            self.assertEqual(len(requests), 1)
            transcript = requests[0][1].content or ""
            self.assertIn("[用户]", transcript)
            self.assertIn("[工具调用] read_file", transcript)
            self.assertIn("[工具结果]", transcript)
            self.assertNotIn("当前问题", transcript)

            session = loop.sessions.get_or_create("sdk:u")
            self.assertEqual(session.summary, provider.summary)
            self.assertGreater(session.summary_messages, 0)
            non_system = [m for m in session.messages if m.role != "system"]
            # 最近 2 个完整回合 + 本轮 user/assistant。
            self.assertEqual(len(non_system), 6)

            saved = Path(tmp) / f"{safe_key('sdk:u')}.jsonl"
            self.assertEqual(len(saved.read_text(encoding="utf-8").splitlines()), 7)

    async def test_summary_failure_keeps_history_and_turn_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _BrokenSummaryProvider()
            loop = _make_loop(
                tmp,
                provider,
                prompt_max_tokens=120,
                context_compact_ratio=1.0,
                context_keep_turns=2,
            )
            await _seed_turns(loop, count=8)
            before = len(loop.sessions.get_or_create("sdk:u").messages)
            response = await loop.handle_text(
                channel="sdk", sender="u", text="继续"
            )
            self.assertEqual(response.text, "收到")
            session = loop.sessions.get_or_create("sdk:u")
            self.assertEqual(session.summary, "")
            self.assertGreater(len(session.messages), before)

    async def test_compact_and_status_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _SummaryProvider()
            loop = _make_loop(
                tmp,
                provider,
                prompt_max_tokens=10_000,
                context_compact_ratio=1.0,
                context_keep_turns=1,
            )
            await _seed_turns(loop, count=4)

            compacted = await loop.handle_text(
                channel="sdk", sender="u", text="/compact"
            )
            self.assertIn("已压缩上下文", compacted.text)
            session = loop.sessions.get_or_create("sdk:u")
            self.assertEqual(session.summary, provider.summary)

            status = await loop.handle_text(
                channel="sdk", sender="u", text="/status"
            )
            self.assertIn("会话：sdk:u", status.text)
            self.assertIn("估算 token", status.text)
            self.assertIn("历史摘要：已生成", status.text)


class ContextBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_and_extra_system_are_injected(self) -> None:
        builder = ContextBuilder(
            "SYS", max_chars=100_000, max_tokens=10_000, workspace="."
        )
        prompt = await builder.abuild(
            "sdk:u",
            [Message(role="user", content="最近的问题")],
            summary="更早的摘要内容",
            extra_system="## 技能说明\n临时规则",
        )
        self.assertIn("更早的摘要内容", prompt[0].content or "")
        self.assertIn("临时规则", prompt[0].content or "")
        self.assertEqual(prompt[-1].content, "最近的问题")

    async def test_token_budget_drops_oldest_turns(self) -> None:
        builder = ContextBuilder(
            "SYS", max_chars=100_000, max_tokens=120, workspace="."
        )
        history = []
        for index in range(20):
            history.append(Message(role="user", content=f"问题{index}：" + "内容" * 40))
            history.append(Message(role="assistant", content="回答" * 40))
        prompt = await builder.abuild("sdk:u", history)
        self.assertLess(len(prompt), len(history) + 1)
        self.assertEqual(prompt[-1].role, "assistant")
        self.assertIn("问题19", prompt[-2].content or "")

    async def test_tool_schema_and_reserve_shrink_history_budget(self) -> None:
        history = []
        for index in range(6):
            history.append(Message(role="user", content=f"问题{index}：" + "内容" * 30))
            history.append(Message(role="assistant", content="回答" * 30))

        plain = ContextBuilder(
            "SYS", max_chars=100_000, max_tokens=600, reserve_tokens=0
        )
        priced = ContextBuilder(
            "SYS",
            max_chars=100_000,
            max_tokens=600,
            tool_tokens=300,
            reserve_tokens=100,
        )
        plain_prompt = await plain.abuild("sdk:u", history)
        priced_prompt = await priced.abuild("sdk:u", history)

        self.assertLess(len(priced_prompt), len(plain_prompt))
        self.assertEqual(priced.overhead_tokens() - plain.overhead_tokens(), 400)
        # 裁剪只能整回合进行：user 与 assistant 必须成对，不能拆散调用链。
        roles = [message.role for message in priced_prompt[1:]]
        self.assertEqual(roles.count("user"), roles.count("assistant"))
        self.assertEqual(priced_prompt[-1].role, "assistant")
        self.assertGreater(priced.total_tokens(priced_prompt), 0)

    async def test_abuild_passes_latest_user_question_to_memory(self) -> None:
        calls: list[tuple[str, str]] = []

        class _Backend:
            enabled = True

            def render(self, key: str, max_chars: int, query: str = "") -> str:
                calls.append((key, query))
                return ""

        builder = ContextBuilder("SYS", _Backend(), max_chars=100_000)
        await builder.abuild(
            "sdk:u",
            [
                Message(role="user", content="旧问题"),
                Message(role="assistant", content="旧回答"),
                Message(role="user", content="当前问题"),
            ],
        )
        self.assertEqual(calls, [("sdk:u", "当前问题")])


if __name__ == "__main__":
    unittest.main()
