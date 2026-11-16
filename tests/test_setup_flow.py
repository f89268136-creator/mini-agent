"""首启配置链路离线测试：分层配置、凭据、预检、端口与设置页令牌。

全部离线运行：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python -m unittest \
        mini_agent.tests.test_setup_flow -v
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import socket
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from mini_agent.config import (
    ConfigError,
    ConfigIssue,
    ConfigReport,
    MiniConfig,
    config_diff,
    load_config,
    load_config_report,
    load_file_config,
    write_config,
)
from mini_agent.runtime.preflight import (
    BLOCKED,
    READY,
    SETUP_REQUIRED,
    PortUnavailableError,
    choose_port,
    run_preflight,
    service_mode,
)
from mini_agent.sdk import MiniBot
from mini_agent.utils import credentials

DASHSCOPE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LOCAL_V1 = "http://127.0.0.1:11434/v1"


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)


class _FakeKeyring:
    """最小钥匙串替身：不接触真实系统凭据库。"""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_keyring(self):
        return self

    def get_password(self, service: str, username: str):
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        self.store[(service, username)] = value

    def delete_password(self, service: str, username: str) -> None:
        self.store.pop((service, username), None)


class ConfigLayeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_path = os.path.join(self.tmp.name, "user.json")
        self.project_path = os.path.join(self.tmp.name, "project.json")
        for name, value in (
            ("MINI_CONFIG", None),
            ("MODEL", None),
            ("OPENAI_API_KEY", None),
            ("MINI_CREDENTIALS_FILE", None),
        ):
            patcher = mock.patch.dict(os.environ, {name: value} if value else {})
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("MINI_CONFIG", None)
        os.environ.pop("MODEL", None)
        os.environ.pop("OPENAI_API_KEY", None)
        for target, value in (
            ("mini_agent.config.user_config_path", lambda: self.user_path),
            ("mini_agent.config.project_config_path", lambda: self.project_path),
        ):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch(
            "mini_agent.utils.credentials.load_api_key", return_value=(None, "")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_precedence_user_then_project_then_env(self) -> None:
        _write_json(self.user_path, {"model": "user-model"})
        _write_json(self.project_path, {"model": "project-model"})
        self.assertEqual(load_config_report().config.model, "project-model")
        os.environ["MODEL"] = "env-model"
        self.assertEqual(load_config_report().config.model, "env-model")

    def test_explicit_path_ignores_other_files(self) -> None:
        _write_json(self.user_path, {"model": "user-model"})
        _write_json(self.project_path, {"model": "project-model"})
        report = load_config_report(self.user_path)
        self.assertEqual(report.config.model, "user-model")
        self.assertEqual(report.paths, [self.user_path])

    def test_corrupt_config_enters_repair_mode_instead_of_blocking(self) -> None:
        with open(self.project_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        report = load_config_report()
        self.assertTrue(report.needs_repair)
        self.assertIn("JSON", "；".join(report.errors))
        # 仍然能拿到一份可用于启动修复页面的配置。
        self.assertEqual(report.config.model, MiniConfig().model)
        with self.assertRaises(ConfigError):
            load_config()

    def test_diff_only_write_and_no_secret(self) -> None:
        target = os.path.join(self.tmp.name, "saved.json")
        config = MiniConfig(model="qwen-plus", api_key="sk-should-not-be-written")
        path = write_config(config, target)
        self.assertEqual(path, target)
        with open(target, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved, {"model": "qwen-plus"})
        self.assertNotIn("api_key", saved)
        self.assertNotIn("credential_source", config_diff(config))
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(target).st_mode)
            self.assertEqual(mode, 0o600)

    def test_file_layer_ignores_env_and_runtime_overrides(self) -> None:
        _write_json(self.project_path, {"model": "file-model"})
        os.environ["MODEL"] = "env-model"
        runtime = MiniConfig(model="env-model", channels=["web"], web_port=9999)
        base = load_file_config(self.project_path)
        self.assertEqual(base.model, "file-model")
        self.assertEqual(base.channels, ["cli"])
        self.assertEqual(base.web_port, 8767)
        self.assertNotEqual(runtime.web_port, base.web_port)

    def test_needs_setup_only_for_remote_without_key(self) -> None:
        self.assertTrue(MiniConfig(api_base=DASHSCOPE).needs_setup())
        self.assertFalse(MiniConfig(api_base=LOCAL_V1).needs_setup())
        self.assertFalse(
            MiniConfig(api_base=DASHSCOPE, api_key="k").needs_setup()
        )
        self.assertFalse(
            MiniConfig(api_base=DASHSCOPE, allow_no_api_key=True).needs_setup()
        )


class CredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file_path = os.path.join(self.tmp.name, "credentials.json")
        patcher = mock.patch.dict(
            os.environ, {"MINI_CREDENTIALS_FILE": self.file_path}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OPENAI_API_KEY", None)
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)

    def test_env_wins_over_keyring_and_file(self) -> None:
        fake = _FakeKeyring()
        fake.set_password(credentials.KEYRING_SERVICE, "api.example.com", "k-store")
        _write_json(self.file_path, {"api.example.com": "k-file"})
        with mock.patch("mini_agent.utils.credentials._keyring", return_value=fake):
            self.assertEqual(
                credentials.load_api_key("https://api.example.com/v1"),
                ("k-store", "keyring"),
            )
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k-env"}):
                self.assertEqual(
                    credentials.load_api_key("https://api.example.com/v1"),
                    ("k-env", "env"),
                )

    def test_file_roundtrip_when_keyring_missing(self) -> None:
        with mock.patch("mini_agent.utils.credentials._keyring", return_value=None):
            self.assertEqual(
                credentials.save_api_key("https://api.example.com/v1", "k-file"),
                "file",
            )
            self.assertEqual(
                credentials.load_api_key("https://api.example.com/v1"),
                ("k-file", "file"),
            )
        if os.name != "nt":
            self.assertEqual(
                stat.S_IMODE(os.stat(self.file_path).st_mode), 0o600
            )

    def test_keyring_preferred_for_save(self) -> None:
        fake = _FakeKeyring()
        with mock.patch("mini_agent.utils.credentials._keyring", return_value=fake):
            self.assertEqual(
                credentials.save_api_key("https://api.example.com/v1", "k-key"),
                "keyring",
            )
            self.assertEqual(
                fake.store[(credentials.KEYRING_SERVICE, "api.example.com")], "k-key"
            )

    def test_missing_backend_error_never_echoes_key(self) -> None:
        with mock.patch("mini_agent.utils.credentials._keyring", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MINI_CREDENTIALS_FILE", None)
                with self.assertRaises(credentials.CredentialsError) as ctx:
                    credentials.save_api_key("https://api.example.com/v1", "sk-secret")
        self.assertNotIn("sk-secret", str(ctx.exception))
        self.assertIn("keyring", str(ctx.exception))

    def test_fail_backend_is_not_reported_available(self) -> None:
        class _FailKeyring:
            class _Backend:
                priority = 0

            def get_keyring(self):
                return self._Backend()

        with mock.patch(
            "mini_agent.utils.credentials._keyring", return_value=_FailKeyring()
        ):
            self.assertFalse(credentials.keyring_available())

    def test_backend_without_priority_is_treated_as_available(self) -> None:
        with mock.patch(
            "mini_agent.utils.credentials._keyring", return_value=_FakeKeyring()
        ):
            self.assertTrue(credentials.keyring_available())


class PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report = ConfigReport(
            config=MiniConfig(
                api_base=DASHSCOPE,
                sessions_dir=os.path.join(self.tmp.name, "sessions"),
                memory_dir=os.path.join(self.tmp.name, "memory"),
                todos_dir=os.path.join(self.tmp.name, "todos"),
            ),
            paths=[],
            issues=[],
        )

    def test_missing_key_is_setup_required(self) -> None:
        result = run_preflight(self.report)
        self.assertEqual(result.status, SETUP_REQUIRED)
        self.assertIn("首次配置", result.setup_reason)

    def test_broken_config_is_repair_not_blocked(self) -> None:
        report = ConfigReport(
            config=self.report.config,
            paths=["broken.json"],
            issues=[ConfigIssue("broken.json", "配置文件不是合法 JSON")],
        )
        result = run_preflight(report)
        self.assertEqual(result.status, SETUP_REQUIRED)
        self.assertTrue(result.repair)
        self.assertFalse(result.ready)

    def test_ready_when_local_endpoint_available(self) -> None:
        report = ConfigReport(
            config=MiniConfig(
                api_base=LOCAL_V1,
                model="qwen2.5:7b",
                sessions_dir=os.path.join(self.tmp.name, "sessions"),
                memory_dir=os.path.join(self.tmp.name, "memory"),
                todos_dir=os.path.join(self.tmp.name, "todos"),
            ),
            paths=[],
            issues=[],
        )
        result = run_preflight(report, check_port=False)
        self.assertEqual(result.status, READY)
        payload = result.as_dict()
        self.assertNotIn("api_key", json.dumps(payload))

    def test_unwritable_state_dir_blocks(self) -> None:
        blocker = os.path.join(self.tmp.name, "blocked")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("not a dir")
        report = ConfigReport(
            config=MiniConfig(
                api_base=LOCAL_V1,
                sessions_dir=blocker,
                memory_dir=os.path.join(self.tmp.name, "memory"),
                todos_dir=os.path.join(self.tmp.name, "todos"),
            ),
            paths=[],
            issues=[],
        )
        result = run_preflight(report)
        self.assertEqual(result.status, BLOCKED)
        self.assertIn("不可写", result.reason)
        self.assertIn("检查目录权限", result.action)

    def test_port_avoidance_is_local_only(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            busy = sock.getsockname()[1]
            port, shifted = choose_port("127.0.0.1", busy, auto=True)
            self.assertTrue(shifted)
            self.assertGreater(port, busy)
            with self.assertRaises(PortUnavailableError):
                choose_port("127.0.0.1", busy, auto=False)

    def test_service_mode_env(self) -> None:
        with mock.patch.dict(os.environ, {"MINI_SERVICE": "1"}):
            self.assertTrue(service_mode())
        with mock.patch.dict(os.environ, {"MINI_SERVICE": "0"}):
            with mock.patch.dict(os.environ, {"INVOCATION_ID": ""}):
                self.assertFalse(service_mode())


class SetupPageTests(unittest.IsolatedAsyncioTestCase):
    """令牌保护 -> 保存 -> 热重配 -> 就绪；全程不访问模型。"""

    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = os.path.join(self.tmp.name, "config.json")
        _write_json(
            self.config_path,
            {
                "model": "qwen-plus",
                "api_base": DASHSCOPE,
                "sessions_dir": os.path.join(self.tmp.name, "sessions"),
                "memory_dir": os.path.join(self.tmp.name, "memory"),
                "todos_dir": os.path.join(self.tmp.name, "todos"),
            },
        )
        patcher = mock.patch.dict(
            os.environ,
            {
                "MINI_CONFIG": self.config_path,
                "MINI_CREDENTIALS_FILE": os.path.join(
                    self.tmp.name, "credentials.json"
                ),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OPENAI_API_KEY", None)
        patcher = mock.patch(
            "mini_agent.utils.credentials.load_api_key", return_value=(None, "")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        config = MiniConfig(
            model="qwen-plus",
            api_base=DASHSCOPE,
            sessions_dir=os.path.join(self.tmp.name, "sessions"),
            memory_dir=os.path.join(self.tmp.name, "memory"),
            todos_dir=os.path.join(self.tmp.name, "todos"),
            channels=["web"],
            web_port=8767,
        )
        self.bot = MiniBot(
            config, allow_setup=True, config_path=self.config_path
        )
        web = self.bot.channels["web"]
        web.port = 0
        await self.bot.start()
        self.web = web
        self.port = web.bound_port

    async def asyncTearDown(self) -> None:
        await self.bot.aclose()

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        token: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-Setup-Token"] = token
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    async def call(self, *args, **kwargs):
        return await asyncio.to_thread(self.request, *args, **kwargs)

    async def test_not_ready_redirects_and_stays_unready(self) -> None:
        status, headers, _ = await self.call("GET", "/")
        self.assertEqual(status, 302)
        self.assertTrue(headers["Location"].endswith("/setup"))
        status, _, _ = await self.call("GET", "/healthz")
        self.assertEqual(status, 200)
        status, _, body = await self.call("GET", "/readyz")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["status"], SETUP_REQUIRED)
        status, _, body = await self.call("GET", "/api/setup")
        data = json.loads(body)
        self.assertTrue(data["setup_token_required"])
        self.assertNotIn("api_key", body.decode())
        self.assertEqual(len(data["providers"]), 3)

    async def test_token_protects_save_and_invalidates_once(self) -> None:
        token = self.web.setup_token
        self.assertTrue(token)

        status, _, _ = await self.call(
            "POST", "/api/setup", {"model": "qwen2.5:7b", "api_base": LOCAL_V1}
        )
        self.assertEqual(status, 403)
        status, _, _ = await self.call(
            "POST",
            "/api/setup",
            {"model": "qwen2.5:7b", "api_base": LOCAL_V1},
            token="wrong",
        )
        self.assertEqual(status, 403)

        status, _, body = await self.call(
            "POST",
            "/api/setup",
            {"model": "qwen2.5:7b", "api_base": LOCAL_V1, "api_key": ""},
            token=token,
        )
        self.assertEqual(status, 200, body)
        result = json.loads(body)
        self.assertTrue(result["ok"])
        self.assertNotIn('"api_key"', body.decode())

        status, _, _ = await self.call("GET", "/readyz")
        self.assertEqual(status, 200)
        status, _, _ = await self.call("GET", "/")
        self.assertEqual(status, 200)

        # 已就绪后设置接口关闭；旧令牌也不再可用。
        status, _, _ = await self.call(
            "POST",
            "/api/setup",
            {"model": "qwen2.5:7b", "api_base": LOCAL_V1},
            token=token,
        )
        self.assertEqual(status, 409)

        with open(self.config_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["model"], "qwen2.5:7b")
        self.assertEqual(saved["api_base"], LOCAL_V1)
        self.assertNotIn("api_key", saved)
        # CLI 的运行时覆盖（channels / web_port）不能落盘。
        self.assertNotIn("channels", saved)
        self.assertNotIn("web_port", saved)
        self.assertEqual(self.bot.config.model, "qwen2.5:7b")
        self.assertEqual(self.web.status()["model"], "qwen2.5:7b")

    async def test_runtime_config_hot_update_and_persistence(self) -> None:
        status, _, body = await self.call("GET", "/api/config")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["enabled"])
        self.assertEqual(data["config"]["max_concurrent_turns"], 8)
        self.assertIn("api_base", data["readonly"])
        self.assertNotIn(b"api_key", body)
        self.assertNotIn(b"sk-", body)

        status, _, body = await self.call(
            "POST",
            "/api/config",
            {"max_concurrent_turns": 2, "stream": False, "memory_enabled": False},
        )
        self.assertEqual(status, 200, body)
        self.assertTrue(json.loads(body)["ok"])
        self.assertEqual(self.bot.config.max_concurrent_turns, 2)
        self.assertFalse(self.bot.config.stream)
        self.assertFalse(self.bot.loop.stream)
        self.assertFalse(self.bot.memory.enabled)

        with open(self.config_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["max_concurrent_turns"], 2)
        self.assertFalse(saved["stream"])
        self.assertFalse(saved["memory_enabled"])
        self.assertNotIn("channels", saved)
        self.assertNotIn("web_port", saved)
        self.assertNotIn("api_key", saved)

        status, _, body = await self.call("GET", "/api/status")
        self.assertFalse(json.loads(body)["stream"])

        original_iterations = self.bot.config.max_iterations
        for payload in (
            {"api_key": "sk-should-not-pass"},
            {"web_port": 9000},
            {"max_iterations": 0},
        ):
            status, _, _ = await self.call("POST", "/api/config", payload)
            self.assertEqual(status, 400, payload)
        self.assertEqual(self.bot.config.max_iterations, original_iterations)


class _HealthHandler(BaseHTTPRequestHandler):
    """只实现单实例探活需要的 /healthz。"""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        body = json.dumps({"ok": True, "service": "mini_agent"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class InstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "instances" / "web.json"

    def _serve(self) -> tuple[ThreadingHTTPServer, int]:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def shutdown() -> None:
            server.shutdown()
            server.server_close()
            thread.join(2)

        self.addCleanup(shutdown)
        return server, int(server.server_address[1])

    def test_acquire_reuses_healthy_instance(self) -> None:
        from mini_agent.runtime.instance import InstanceLock, probe

        _server, port = self._serve()
        lock, running = InstanceLock.acquire(self.path)
        self.assertIsNotNone(lock)
        self.assertIsNone(running)
        lock.write(port=port, url=f"http://127.0.0.1:{port}", version="9.9.9")

        second, running = InstanceLock.acquire(self.path)
        self.assertIsNone(second)
        self.assertIsNotNone(running)
        self.assertEqual(running.port, port)
        self.assertEqual(running.url, f"http://127.0.0.1:{port}")
        self.assertTrue(probe(port))

        lock.release()
        self.assertFalse(self.path.exists())

    def test_stale_record_is_taken_over(self) -> None:
        from mini_agent.runtime.instance import InstanceLock

        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            str(self.path),
            {
                "service": "mini_agent",
                "pid": 999999,
                "port": _free_port(),
                "url": "http://127.0.0.1:1",
            },
        )
        lock, running = InstanceLock.acquire(self.path, wait=0.2)
        self.assertIsNotNone(lock)
        self.assertIsNone(running)
        lock.release()
        self.assertFalse(self.path.exists())

    def test_instance_file_is_scoped_and_overridable(self) -> None:
        from mini_agent.runtime.instance import instance_path

        with mock.patch.dict(os.environ, {"MINI_INSTANCE_FILE": ""}):
            first = instance_path("C:/project-a")
            second = instance_path("C:/project-b")
            self.assertNotEqual(first, second)
        with mock.patch.dict(
            os.environ, {"MINI_INSTANCE_FILE": str(self.path)}
        ):
            self.assertEqual(instance_path("C:/project-a"), self.path)


class CliEntryTests(unittest.TestCase):
    def test_command_resolution(self) -> None:
        from mini_agent.__main__ import _resolve_command, build_parser

        parser = build_parser()
        self.assertEqual(_resolve_command(parser.parse_args([])), "start")
        self.assertEqual(_resolve_command(parser.parse_args(["chat"])), "chat")
        self.assertEqual(_resolve_command(parser.parse_args(["login"])), "login")
        self.assertEqual(_resolve_command(parser.parse_args(["--doctor"])), "doctor")
        self.assertEqual(_resolve_command(parser.parse_args(["init"])), "init")
        import io
        from contextlib import redirect_stderr

        with redirect_stderr(io.StringIO()):
            self.assertIsNone(_resolve_command(parser.parse_args(["web", "-m", "hi"])))

    def test_non_interactive_without_credentials_exits_2(self) -> None:
        from mini_agent.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            _write_json(
                config_path,
                {
                    "model": "qwen-plus",
                    "api_base": DASHSCOPE,
                    "sessions_dir": os.path.join(tmp, "sessions"),
                    "memory_dir": os.path.join(tmp, "memory"),
                    "todos_dir": os.path.join(tmp, "todos"),
                },
            )
            env = {
                "MINI_CONFIG": config_path,
                "MINI_CREDENTIALS_FILE": os.path.join(tmp, "credentials.json"),
            }
            with mock.patch.dict(os.environ, env):
                os.environ.pop("OPENAI_API_KEY", None)
                with mock.patch(
                    "mini_agent.utils.credentials.load_api_key", return_value=(None, "")
                ):
                    import io
                    from contextlib import redirect_stderr

                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        code = main(["chat", "-m", "hi"])
            self.assertEqual(code, 2)
            self.assertIn("钥匙串", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
