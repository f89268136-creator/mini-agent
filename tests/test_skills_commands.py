"""技能包与可注册斜杠命令的回归测试。

全部离线运行：
    .venv/Scripts/python.exe -m unittest mini_agent.tests.test_skills_commands -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_agent.agent.commands import CommandRegistry
from mini_agent.agent.context import ContextBuilder
from mini_agent.agent.loop import AgentLoop
from mini_agent.agent.runner import AgentRunner
from mini_agent.agent.skills import SkillRegistry
from mini_agent.bus import MessageBus
from mini_agent.providers.base import LLMProvider, LLMResponse, Message
from mini_agent.session import SessionManager


class _CaptureProvider(LLMProvider):
    def __init__(self) -> None:
        self.requests: list[list] = []

    async def chat(self, messages, tools=None, on_delta=None):
        self.requests.append(list(messages))
        return LLMResponse(content="做好啦")


def _write_skill(root: Path, folder: str, text: str) -> None:
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(text, encoding="utf-8")


def _make_loop(tmp: str, provider, skills: SkillRegistry | None) -> AgentLoop:
    context = ContextBuilder(
        "SYS",
        max_chars=100_000,
        workspace=tmp,
        skills=skills,
    )
    return AgentLoop(
        MessageBus(),
        AgentRunner(provider, max_iterations=2),
        SessionManager(tmp, "SYS", history_limit=50),
        session_idle_timeout=0,
        context=context,
        prompt_max_tokens=0,
        skills=skills,
    )


class CommandRegistryTests(unittest.TestCase):
    def test_register_parse_aliases_and_help(self) -> None:
        registry = CommandRegistry()

        async def handler(ctx):
            return ctx.args

        registry.register(
            "demo", handler, description="演示", usage="/demo", aliases=("d",)
        )
        self.assertEqual(registry.parse("/demo a b"), ("demo", "a b"))
        self.assertEqual(registry.parse("hello"), None)
        self.assertEqual(registry.get("d").name, "demo")
        self.assertIn("/demo", registry.help_text())
        with self.assertRaises(ValueError):
            registry.register("bad name", handler)


class SkillRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.user_root = self.root / "user_skills"
        self.project_root = self.root / "project_skills"
        _write_skill(
            self.user_root,
            "review",
            "---\nname: review\ndescription: 用户级审查\n---\n用户版正文",
        )
        _write_skill(
            self.project_root,
            "review",
            "---\nname: review\ndescription: 项目级审查\n---\n项目版正文",
        )
        (self.project_root / "notes.md").write_text(
            "---\nname: notes\ndescription: 快速笔记\n---\n笔记正文\n",
            encoding="utf-8",
        )
        _write_skill(
            self.project_root,
            "bad",
            "---\nname: bad/name\ndescription: 非法\n---\n正文",
        )
        self.registry = SkillRegistry(
            [("user", self.user_root), ("project", self.project_root)]
        )

    def test_project_skill_overrides_user_and_bad_names_are_skipped(self) -> None:
        self.assertEqual(self.registry.scan(), 2)
        review = self.registry.get("review")
        self.assertEqual(review.source, "project")
        self.assertEqual(review.description, "项目级审查")
        self.assertIn("项目版正文", review.load_body())
        self.assertIsNone(self.registry.get("bad"))

        catalog = self.registry.catalog()
        self.assertIn("review（项目）", catalog)
        self.assertIn("notes（项目）", catalog)
        self.assertNotIn("项目版正文", catalog)

    def test_body_loads_lazily_from_disk(self) -> None:
        self.registry.scan()
        path = self.project_root / "review" / "SKILL.md"
        path.write_text(
            "---\nname: review\ndescription: 项目级审查\n---\n更新后的正文",
            encoding="utf-8",
        )
        self.assertIn("更新后的正文", self.registry.get("review").load_body())

    def test_disabled_registry_is_empty(self) -> None:
        registry = SkillRegistry(
            [("user", self.user_root)], enabled=False
        )
        self.assertEqual(registry.scan(), 0)
        self.assertEqual(registry.catalog(), "")


class SkillAndCommandLoopTests(unittest.IsolatedAsyncioTestCase):
    def _skills(self, root: Path) -> SkillRegistry:
        registry = SkillRegistry([("project", root)])
        registry.scan()
        return registry

    async def test_builtin_alias_and_custom_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _CaptureProvider()
            loop = _make_loop(tmp, provider, skills=None)
            self.assertEqual(loop.commands.get("clear").name, "reset")
            self.assertEqual(loop.commands.get("?").name, "help")

            async def echo(ctx):
                return f"echo:{ctx.args}"

            loop.commands.register("echo", echo, description="回显")
            reply = await loop.handle_text(channel="sdk", sender="u", text="/echo hi")
            self.assertEqual(reply.text, "echo:hi")
            self.assertEqual(provider.requests, [])

    async def test_skill_invocation_injects_transient_system_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            _write_skill(
                root,
                "review",
                "---\nname: review\ndescription: 代码审查\n---\n审查时先看测试覆盖。",
            )
            provider = _CaptureProvider()
            loop = _make_loop(tmp, provider, skills=self._skills(root))

            reply = await loop.handle_text(
                channel="sdk", sender="u", text="/review 只看 src 目录"
            )
            self.assertEqual(reply.text, "做好啦")
            system_text = provider.requests[-1][0].content or ""
            self.assertIn("当前技能：review", system_text)
            self.assertIn("审查时先看测试覆盖", system_text)

            session = loop.sessions.get_or_create("sdk:u")
            self.assertEqual(
                [message.role for message in session.messages],
                ["system", "user", "assistant"],
            )
            self.assertEqual(session.messages[1].content, "只看 src 目录")
            self.assertNotIn("审查时先看测试覆盖", session.messages[1].content or "")

    async def test_skill_command_without_args_uses_default_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            _write_skill(
                root,
                "notes",
                "---\nname: notes\ndescription: 笔记\n---\n整理成要点。",
            )
            provider = _CaptureProvider()
            loop = _make_loop(tmp, provider, skills=self._skills(root))
            await loop.handle_text(channel="sdk", sender="u", text="/skill notes")
            session = loop.sessions.get_or_create("sdk:u")
            self.assertIn("技能「notes」", session.messages[1].content or "")

    async def test_skills_listing_reload_and_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            _write_skill(
                root,
                "review",
                "---\nname: review\ndescription: 代码审查\n---\n正文",
            )
            provider = _CaptureProvider()
            loop = _make_loop(tmp, provider, skills=self._skills(root))

            listed = await loop.handle_text(channel="sdk", sender="u", text="/skills")
            self.assertIn("review（项目）", listed.text)

            _write_skill(
                root,
                "writer",
                "---\nname: writer\ndescription: 写作\n---\n正文",
            )
            reloaded = await loop.handle_text(
                channel="sdk", sender="u", text="/skills reload"
            )
            self.assertIn("2 个技能", reloaded.text)

            unknown = await loop.handle_text(
                channel="sdk", sender="u", text="/missing"
            )
            self.assertIn("未知命令", unknown.text)
            self.assertEqual(provider.requests, [])

    async def test_disabled_skills_fall_back_to_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _CaptureProvider()
            registry = SkillRegistry([], enabled=False)
            registry.scan()
            loop = _make_loop(tmp, provider, skills=registry)
            reply = await loop.handle_text(channel="sdk", sender="u", text="/skills")
            self.assertIn("已禁用", reply.text)
            reply = await loop.handle_text(channel="sdk", sender="u", text="/review")
            self.assertIn("未知命令", reply.text)

    async def test_context_injects_catalog_but_not_skill_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            _write_skill(
                root,
                "review",
                "---\nname: review\ndescription: 代码审查\n---\n不应出现在目录里的正文",
            )
            builder = ContextBuilder(
                "SYS", workspace=tmp, skills=self._skills(root)
            )
            prompt = await builder.abuild(
                "sdk:u", [Message(role="user", content="hi")]
            )
            system_text = prompt[0].content or ""
            self.assertIn("review（项目）", system_text)
            self.assertNotIn("不应出现在目录里的正文", system_text)


if __name__ == "__main__":
    unittest.main()
