"""Web 渠道离线测试：HTTP 接口、SSE、安全校验与配置。

不访问模型、不发起外网请求；服务器绑定 127.0.0.1 的随机端口。
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace

from mini_agent.bus import MessageBus, OutboundMessage
from mini_agent.channels import WebChannel, build_channels, validate_channels
from mini_agent.config import ConfigError, MiniConfig
from mini_agent.memory import MemoryStore
from mini_agent.session.todo import TodoStore


class _FakeSessions:
    """只实现历史查询、会话列表与会话删除需要的接口。"""

    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}
        self.deleted: list[str] = []

    def get_or_create(self, key: str) -> SimpleNamespace:
        if key not in self.records:
            self.records[key] = SimpleNamespace(messages=[])
        return self.records[key]

    def list_sessions(self, prefix: str = "") -> list[dict[str, object]]:
        items = []
        for key, record in self.records.items():
            if not key.startswith(prefix):
                continue
            first_user = next(
                (m for m in record.messages if m.role == "user" and m.content), None
            )
            items.append(
                {
                    "key": key,
                    "title": (first_user.content[:20] if first_user else "新对话"),
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "messages": len(record.messages),
                }
            )
        return items

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.records.pop(key, None)


class _FakeSubagents:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel_session(self, key: str) -> int:
        self.cancelled.append(key)
        return 1


class _FakeLoop:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.running: set[str] = set()
        self.subagents = _FakeSubagents()

    async def cancel_turn(self, key: str) -> bool:
        self.cancelled.append(key)
        return True

    def is_turn_active(self, key: str) -> bool:
        return key in self.running


class _FakeKnowledge:
    def __init__(self) -> None:
        self.docs: dict[str, list[str]] = {}
        self.reindexed = 0
        self.last_persist_error: str | None = None

    def stats(self) -> dict[str, object]:
        return {
            "sources": len(self.docs),
            "chunks": sum(len(items) for items in self.docs.values()),
            "chars": sum(len(text) for items in self.docs.values() for text in items),
            "needs_reindex": False,
        }

    def list_sources(self) -> list[dict[str, object]]:
        return [
            {
                "source": name,
                "chunks": len(items),
                "chars": sum(len(text) for text in items),
                "updated_at": "",
            }
            for name, items in self.docs.items()
        ]

    async def ingest(self, content: str, source: str = "", replace_source: bool = False) -> int:
        self.docs[source] = [content]
        return 1

    async def delete_source(self, source: str) -> int:
        return 1 if self.docs.pop(source, None) is not None else 0

    async def reindex(self, force: bool = False) -> int:
        self.reindexed += 1
        return sum(len(items) for items in self.docs.values())


class WebChannelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus = MessageBus()
        self.sessions = _FakeSessions()
        self.memory = MemoryStore(os.path.join(self.tmp.name, "memory"))
        self.todos = TodoStore(os.path.join(self.tmp.name, "todos"))
        self.knowledge = _FakeKnowledge()
        self.loop_ref = _FakeLoop()
        self.config_applied: list[dict] = []

        def config_snapshot() -> dict:
            return {
                "enabled": True,
                "config": {
                    "max_concurrent_turns": 8,
                    "stream": True,
                    "tool_approval": "ask",
                },
                "readonly": {
                    "api_base": "http://127.0.0.1:8000/v1",
                    "credential_source": "env",
                },
            }

        async def apply_config(payload: dict) -> dict:
            if "api_key" in payload:
                raise ConfigError("端点与密钥请使用首启设置页或 mini-agent login 修改")
            if "web_port" in payload:
                raise ConfigError("不支持在运行时修改：web_port")
            if payload.get("boom"):
                raise RuntimeError("运行时配置接口未启用")
            self.config_applied.append(dict(payload))
            return {
                "ok": True,
                "config_path": "/tmp/config.json",
                "applied": sorted(payload),
                "config": {**config_snapshot()["config"], **payload},
            }

        self.channel = WebChannel(self.bus, port=0)
        self.channel.configure(
            sessions=self.sessions,
            memory=self.memory,
            todos=self.todos,
            knowledge=self.knowledge,
            loop=self.loop_ref,
            model="test-model",
            stream=True,
            max_input_chars=1000,
            version="9.9.9",
            approval_timeout=2.0,
            policy_info={"tool_approval": "ask"},
            disclosure={"model_endpoint": "http://127.0.0.1:8000/v1"},
            config_snapshot=config_snapshot,
            on_config=apply_config,
        )
        await self.channel.start()
        self.port = self.channel.bound_port
        self.assertTrue(self.port > 0)

    async def asyncTearDown(self) -> None:
        await self.channel.stop()
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            merged = {"Content-Type": "application/json", **(headers or {})}
            connection.request(method, path, body=payload, headers=merged)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    async def call(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        return await asyncio.to_thread(self.request, method, path, body, headers)

    # ------------------------------------------------------------------
    # 静态页与状态
    # ------------------------------------------------------------------
    async def test_index_and_status(self) -> None:
        status, headers, body = await self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("mini_agent", body.decode("utf-8"))
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

        status, _, body = await self.call("GET", "/api/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["model"], "test-model")
        self.assertEqual(data["version"], "9.9.9")
        self.assertTrue(data["memory_enabled"])
        self.assertTrue(data["todos_enabled"])
        self.assertTrue(data["loopback_only"])

    async def test_unknown_path_returns_404(self) -> None:
        status, _, body = await self.call("GET", "/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body))

    # ------------------------------------------------------------------
    # 入站消息
    # ------------------------------------------------------------------
    async def test_post_message_publishes_inbound(self) -> None:
        status, _, body = await self.call(
            "POST", "/api/message", {"session": "s1", "text": "你好"}
        )
        self.assertEqual(status, 202)
        self.assertTrue(json.loads(body)["ok"])

        inbound = await asyncio.wait_for(self.bus.consume_inbound(), timeout=2)
        self.assertEqual(inbound.channel, "web")
        self.assertEqual(inbound.sender, "s1")
        self.assertEqual(inbound.session_key, "web:s1")
        self.assertEqual(inbound.text, "你好")

    async def test_invalid_payloads_are_rejected(self) -> None:
        cases = [
            ("POST", "/api/message", {"session": "../etc", "text": "x"}, 400),
            ("POST", "/api/message", {"session": "s1", "text": ""}, 400),
            ("POST", "/api/message", {"session": "s1", "text": "x" * 1001}, 413),
            ("GET", "/api/history?session=bad!", None, 400),
            ("GET", "/api/memory?session=", None, 400),
        ]
        for method, path, body, expected in cases:
            status, _, _ = await self.call(method, path, body)
            self.assertEqual(status, expected, f"{method} {path}")
        self.assertEqual(self.bus.inbound_size(), 0)

    async def test_oversized_body_is_rejected(self) -> None:
        status, _, _ = await self.call(
            "POST",
            "/api/message",
            {"session": "s1", "text": "x" * (130 * 1024)},
        )
        self.assertEqual(status, 413)
        self.assertEqual(self.bus.inbound_size(), 0)

    # ------------------------------------------------------------------
    # 历史 / 记忆 / 清单
    # ------------------------------------------------------------------
    async def test_history_filters_non_chat_messages(self) -> None:
        record = self.sessions.get_or_create("web:s1")
        record.messages = [
            SimpleNamespace(role="system", content="人设"),
            SimpleNamespace(role="user", content="你好"),
            SimpleNamespace(role="assistant", content=None),
            SimpleNamespace(role="tool", content="工具结果"),
            SimpleNamespace(role="assistant", content="在的"),
        ]
        status, _, body = await self.call("GET", "/api/history?session=s1")
        self.assertEqual(status, 200)
        messages = json.loads(body)["messages"]
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "在的"},
            ],
        )

    async def test_memory_and_todos_endpoints(self) -> None:
        self.memory.add("web:s1", "喜欢 Python")
        self.memory.add("web:s1", "正在学习 Agent 框架")
        self.todos.write(
            "web:s1",
            [
                {"content": "读完 loop.py", "status": "completed"},
                {"content": "写一个工具", "status": "in_progress"},
            ],
        )

        status, _, body = await self.call("GET", "/api/memory?session=s1")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["enabled"])
        self.assertEqual(data["entries"][0]["text"], "正在学习 Agent 框架")
        self.assertEqual(len(data["entries"]), 2)

        status, _, body = await self.call("GET", "/api/todos?session=s1")
        self.assertEqual(status, 200)
        items = json.loads(body)["items"]
        self.assertEqual([item["status"] for item in items], ["completed", "in_progress"])

        # 会话隔离：另一个 session 看不到上面的数据
        status, _, body = await self.call("GET", "/api/memory?session=s2")
        self.assertEqual(json.loads(body)["entries"], [])

    async def test_history_includes_turn_metadata(self) -> None:
        record = self.sessions.get_or_create("web:s1")
        record.messages = [
            SimpleNamespace(role="user", content="文档里说了什么？"),
            SimpleNamespace(
                role="assistant",
                content="说了三件事。",
                metadata={
                    "citations": [
                        {
                            "source": "notes.md",
                            "start_line": 3,
                            "end_line": 5,
                            "score": 0.8,
                            "snippet": "三件事……",
                        }
                    ],
                    "tools": [{"id": "t1", "name": "knowledge_search", "status": "done"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                },
            ),
        ]
        status, _, body = await self.call("GET", "/api/history?session=s1")
        self.assertEqual(status, 200)
        messages = json.loads(body)["messages"]
        self.assertEqual(messages[1]["citations"][0]["source"], "notes.md")
        self.assertEqual(messages[1]["tools"][0]["name"], "knowledge_search")
        self.assertEqual(messages[1]["usage"]["total_tokens"], 18)

    async def test_sessions_endpoint_and_delete(self) -> None:
        record = self.sessions.get_or_create("web:s1")
        record.messages = [SimpleNamespace(role="user", content="你好")]
        self.sessions.get_or_create("cli:local").messages = [
            SimpleNamespace(role="user", content="不该出现")
        ]
        self.memory.add("web:s1", "记住我")
        self.todos.write("web:s1", [{"content": "任务", "status": "pending"}])

        status, _, body = await self.call("GET", "/api/sessions")
        self.assertEqual(status, 200)
        sessions = json.loads(body)["sessions"]
        self.assertEqual([item["session"] for item in sessions], ["s1"])
        self.assertEqual(sessions[0]["messages"], 1)
        self.assertFalse(sessions[0]["running"])

        self.loop_ref.running.add("web:s1")
        status, _, body = await self.call("GET", "/api/sessions")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["sessions"][0]["running"])
        self.loop_ref.running.clear()

        status, _, body = await self.call(
            "POST", "/api/session/delete", {"session": "s1"}
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertTrue(result["ok"])
        self.assertEqual(result["memory_removed"], 1)
        self.assertEqual(result["todos_removed"], 1)
        self.assertIn("web:s1", self.sessions.deleted)
        self.assertIn("web:s1", self.loop_ref.cancelled)
        self.assertIn("web:s1", self.loop_ref.subagents.cancelled)
        self.assertEqual(self.memory.entries("web:s1"), [])
        self.assertEqual(self.todos.items("web:s1"), [])

    async def test_stop_endpoint_cancels_active_turn(self) -> None:
        status, _, body = await self.call("POST", "/api/stop", {"session": "s1"})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["stopped"])
        self.assertEqual(self.loop_ref.cancelled, ["web:s1"])
        status, _, _ = await self.call("POST", "/api/stop", {"session": "../bad"})
        self.assertEqual(status, 400)

    async def test_quit_endpoint(self) -> None:
        status, _, body = await self.call("GET", "/api/status")
        self.assertFalse(json.loads(body)["can_quit"])
        status, _, _ = await self.call("POST", "/api/quit", {"confirm": True})
        self.assertEqual(status, 501)

        stopped = threading.Event()
        self.channel.set_quit_handler(stopped.set)
        status, _, _ = await self.call("POST", "/api/quit", {"confirm": False})
        self.assertEqual(status, 400)
        self.assertFalse(stopped.is_set())

        status, _, body = await self.call("POST", "/api/quit", {"confirm": True})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertTrue(await asyncio.to_thread(stopped.wait, 2))

        status, _, body = await self.call("GET", "/api/status")
        self.assertTrue(json.loads(body)["can_quit"])

    async def test_approval_flow_over_sse_and_http(self) -> None:
        stream = self.channel.subscribe("s1")
        self.assertIsNotNone(stream)
        task = asyncio.create_task(
            self.channel.request_approval(
                session_key="web:s1",
                sender="s1",
                tool_name="write_file",
                arguments={"path": "a.txt"},
            )
        )
        event = await asyncio.wait_for(asyncio.to_thread(stream.get, 2), timeout=3)
        self.assertEqual(event["type"], "approval")
        self.assertEqual(event["session"], "s1")
        self.assertEqual(event["tool"], "write_file")
        self.assertIn("a.txt", event["arguments"])

        status, _, body = await self.call(
            "POST",
            "/api/approval",
            {"approval_id": event["approval_id"], "approved": True},
        )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertTrue(await asyncio.wait_for(task, timeout=2))

        status, _, body = await self.call(
            "POST", "/api/approval", {"approval_id": "missing", "approved": True}
        )
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["ok"])

    async def test_knowledge_endpoints(self) -> None:
        status, _, body = await self.call("GET", "/api/knowledge")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["enabled"])

        status, _, body = await self.call(
            "POST",
            "/api/knowledge/import",
            {"name": "notes.md", "content": "# 标题\n\n正文内容"},
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["source"], "notes.md")

        status, _, body = await self.call(
            "POST", "/api/knowledge/import", {"name": "../evil.md", "content": "x"}
        )
        self.assertEqual(status, 400)
        status, _, _ = await self.call(
            "POST", "/api/knowledge/import", {"name": "a.md", "content": ""}
        )
        self.assertEqual(status, 400)

        status, _, body = await self.call("GET", "/api/knowledge")
        data = json.loads(body)
        self.assertEqual(data["stats"]["sources"], 1)
        self.assertEqual(data["sources"][0]["source"], "notes.md")

        status, _, body = await self.call(
            "POST", "/api/knowledge/delete", {"source": "notes.md"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["removed"], 1)

        status, _, body = await self.call("POST", "/api/knowledge/reindex", {})
        self.assertEqual(status, 200)
        self.assertEqual(self.knowledge.reindexed, 1)

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------
    async def test_sse_delivers_delta_and_final(self) -> None:
        ready = threading.Event()
        events: list[dict] = []

        def reader() -> None:
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=5
            )
            try:
                connection.request("GET", "/api/stream?session=s1")
                response = connection.getresponse()
                while len(events) < 3:
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[len(b"data: ") :])
                    events.append(event)
                    if event.get("type") == "ready":
                        ready.set()
            finally:
                connection.close()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        self.assertTrue(await asyncio.to_thread(ready.wait, 3), "SSE 未及时就绪")

        await self.channel.send(
            OutboundMessage(channel="web", to="s1", text="你", partial=True)
        )
        await self.channel.send(
            OutboundMessage(channel="web", to="s1", text="你好", streamed=True)
        )
        await asyncio.to_thread(thread.join, 5)
        self.assertEqual([event["type"] for event in events[:3]], ["ready", "delta", "final"])
        self.assertEqual(events[1]["session"], "s1")
        self.assertEqual(events[2]["session"], "s1")
        self.assertEqual(events[1]["text"], "你")
        self.assertEqual(events[2]["text"], "你好")
        self.assertTrue(events[2]["streamed"])

    async def test_config_endpoint_reads_and_applies(self) -> None:
        status, _, body = await self.call("GET", "/api/config")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["enabled"])
        self.assertEqual(data["config"]["max_concurrent_turns"], 8)
        self.assertIn("api_base", data["readonly"])
        self.assertNotIn(b"api_key", body)
        self.assertNotIn(b"secret", body.lower())

        status, _, body = await self.call(
            "POST", "/api/config", {"max_concurrent_turns": 3, "stream": False}
        )
        self.assertEqual(status, 200, body)
        result = json.loads(body)
        self.assertTrue(result["ok"])
        self.assertEqual(result["applied"], ["max_concurrent_turns", "stream"])
        self.assertEqual(self.config_applied, [{"max_concurrent_turns": 3, "stream": False}])

        status, _, body = await self.call(
            "POST", "/api/config", {"api_key": "sk-should-not-pass"}
        )
        self.assertEqual(status, 400)
        self.assertIn("密钥", json.loads(body)["error"])
        status, _, _ = await self.call("POST", "/api/config", {"web_port": 9000})
        self.assertEqual(status, 400)
        status, _, _ = await self.call("POST", "/api/config", {"boom": True})
        self.assertEqual(status, 503)
        self.assertEqual(len(self.config_applied), 1)

    async def test_sse_unsubscribes_after_stop(self) -> None:
        stream = self.channel.subscribe("s1")
        self.assertIsNotNone(stream)
        self.channel.unsubscribe("s1", stream)
        self.assertEqual(self.channel._subs, {})

    # ------------------------------------------------------------------
    # 来源校验
    # ------------------------------------------------------------------
    async def test_origin_and_host_guards(self) -> None:
        status, _, _ = await self.call(
            "GET", "/api/status", headers={"Origin": "http://evil.example"}
        )
        self.assertEqual(status, 403)
        status, _, _ = await self.call(
            "GET", "/api/status", headers={"Host": "evil.example"}
        )
        self.assertEqual(status, 403)
        status, _, _ = await self.call(
            "GET", "/api/status", headers={"Origin": f"http://127.0.0.1:{self.port}"}
        )
        self.assertEqual(status, 200)

    async def test_non_loopback_host_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WebChannel(self.bus, port=0, host="0.0.0.0")

    # ------------------------------------------------------------------
    # 配置与工厂
    # ------------------------------------------------------------------
    def test_config_web_port(self) -> None:
        config = MiniConfig()
        self.assertEqual(config.web_port, 8767)
        with self.assertRaises(ConfigError):
            MiniConfig(web_port=0)
        with self.assertRaises(ConfigError):
            MiniConfig(channels=["echo", "web"], echo_port=9000, web_port=9000)

    def test_build_channels_includes_web(self) -> None:
        validate_channels(["cli", "web"])
        channels = build_channels(["web"], self.bus, web_port=8799)
        self.assertIsInstance(channels["web"], WebChannel)
        self.assertEqual(channels["web"].port, 8799)


if __name__ == "__main__":
    unittest.main()
