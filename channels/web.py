"""Web 渠道：内置单页聊天前端（HTTP + SSE）。

它和 CLI/Echo 一样只是总线的一个"进出口"：浏览器 POST 消息 ->
``publish_inbound``；Agent 的流式增量与最终回复 -> SSE。会话键固定为
``web:<session>``，与 CLI 的 ``cli:local``、Echo 的 ``echo:peer:*`` 完全隔离。

安全边界（刻意保持最小）：

- 只监听回环地址，且校验 Host / Origin，降低 DNS rebinding 风险；
- 没有登录与鉴权。对外提供服务必须由反向代理补上认证与 TLS；
- 页面内容全部经 HTML 转义后再渲染，不执行模型输出里的原始标签；
- 请求体大小、会话 ID、同时打开的流数量都有上限。

协议：

    GET  /                      单页应用（未就绪时 302 到 /setup）
    GET  /setup                 首启设置/修复页
    GET  /healthz               存活探针（只要进程在就返回 200）
    GET  /readyz                就绪探针（未配置完成时 503）
    GET  /api/status            运行时信息（模型、记忆/清单开关等）
    GET  /api/setup             设置页状态（不含密钥）
    POST /api/setup             保存模型/端点/密钥（需要一次性令牌）
    POST /api/setup/test        测试连接（需要一次性令牌，不保存）
    GET  /api/config            可运行时修改的配置项（不含密钥）
    POST /api/config            校验、持久化并热应用配置
    GET  /api/history?session=  会话历史（含引用、工具记录与用量）
    GET  /api/memory?session=   当前会话长期记忆
    GET  /api/todos?session=    当前会话任务清单
    GET  /api/sessions          服务端会话列表（标题、更新时间、消息数、是否运行中）
    GET  /api/knowledge         知识库来源与索引概况
    GET  /api/stream?session=   SSE：ready / delta / tool / approval / final / stopped
    POST /api/message           {session, text} -> 发布一条入站消息
    POST /api/stop              {session} -> 停止当前生成
    POST /api/quit              {confirm: true} -> 请求 CLI 优雅停机
    POST /api/approval          {approval_id, approved} -> 确认/拒绝副作用工具
    POST /api/session/delete    {session} -> 删除会话、记忆与任务清单
    POST /api/knowledge/import  {name, content} -> 导入或替换文档
    POST /api/knowledge/delete  {source} -> 删除文档
    POST /api/knowledge/reindex {} -> 按当前嵌入模型重建向量

设置接口的安全约束：只绑定回环地址；令牌由启动时生成的随机串提供，
经 URL fragment 交给页面（不出现在访问日志里），POST 时放在
``X-Setup-Token`` 头；保存成功后立即失效；密钥不回显、不写日志、不进配置。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import queue
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..bus import InboundMessage, MessageBus, OutboundMessage
from ..config import ConfigError
from ..utils import credentials
from .base import Channel

logger = logging.getLogger(__name__)

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_BODY_BYTES = 128 * 1024
_MAX_UPLOAD_BYTES = 6 * 1024 * 1024
_MAX_UPLOAD_CHARS = 2_000_000
# 超限请求体最多丢弃到这个大小；再大就直接断开，避免被 Content-Length 拖住。
_MAX_DRAIN_BYTES = _MAX_UPLOAD_BYTES + 1024 * 1024
_MAX_APPROVAL_ARGS_CHARS = 4_000
_MAX_EVENT_BACKLOG = 512
_MAX_STREAMS = 64
_MAX_PANEL_ITEMS = 200
_SSE_PING_SECONDS = 15.0
_HISTORY_TIMEOUT = 5.0
_PUBLISH_TIMEOUT = 5.0
_SETUP_TIMEOUT = 40.0
_KNOWLEDGE_TIMEOUT = 120.0
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
_INDEX_FILE = os.path.join(_WEBUI_DIR, "index.html")
_SETUP_FILE = os.path.join(_WEBUI_DIR, "setup.html")

# 首启向导里的服务商模板；只是预填表单，用户仍可改成任意兼容端点。
_PROVIDER_TEMPLATES = [
    {
        "id": "dashscope",
        "label": "阿里云百炼 / DashScope",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-max", "qwen-turbo"],
        "key_url": "https://bailian.console.aliyun.com/",
        "key_hint": "DashScope API-KEY（sk- 开头）",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "key_url": "https://platform.openai.com/api-keys",
        "key_hint": "OpenAI API Key（sk- 开头）",
    },
    {
        "id": "ollama",
        "label": "本地 Ollama",
        "api_base": "http://127.0.0.1:11434/v1",
        "models": ["qwen2.5:7b", "llama3.1:8b"],
        "key_url": "",
        "key_hint": "本地服务通常不校验密钥，可留空",
    },
]


class WebChannel(Channel):
    name = "web"

    def __init__(
        self, bus: MessageBus, port: int = 8767, host: str = "127.0.0.1"
    ) -> None:
        super().__init__(bus)
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "Web 渠道没有登录鉴权，只允许监听回环地址（127.0.0.1 / ::1）"
            )
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._index_bytes = b""
        self._setup_html = b""
        self._stopping = False

        self._sessions = None
        self._memory = None
        self._todos = None
        self._knowledge = None
        self._loop_ref = None
        self._model = ""
        self._stream = True
        self._max_input_chars = 20_000
        self._version = ""
        self._approval_timeout = 120.0
        self._policy_info: dict = {}
        self._disclosure: dict = {}

        # 首启设置：令牌只在启动时生成并通过终端打印的链接交付；
        # preflight/on_setup/on_test 由 MiniBot 注入，保持渠道与 SDK 解耦。
        self._setup_token = ""
        self._preflight = None
        self._on_setup = None
        self._on_test = None
        # 运行时配置：snapshot 只读（HTTP 线程），apply 桥接到事件循环。
        self._config_snapshot = None
        self._on_config = None
        # 停机回调由 CLI 注入；库模式直接使用 WebChannel 时没有该入口。
        self._on_quit = None

        # sender(session id) -> 打开的 SSE 连接队列；同一会话可以开多个标签页。
        self._subs: dict[str, set[queue.Queue]] = {}
        self._subs_lock = threading.Lock()
        # approval_id -> Future；只在事件循环线程里读写。
        self._approvals: dict[str, asyncio.Future[bool]] = {}

    # ------------------------------------------------------------------
    # 装配与生命周期
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        sessions=None,
        memory=None,
        todos=None,
        knowledge=None,
        loop=None,
        model: str = "",
        stream: bool = True,
        max_input_chars: int = 20_000,
        version: str = "",
        approval_timeout: float = 120.0,
        policy_info: dict | None = None,
        disclosure: dict | None = None,
        preflight=None,
        on_setup=None,
        on_test=None,
        config_snapshot=None,
        on_config=None,
    ) -> None:
        """注入只读查询依赖与首启设置回调；由 MiniBot 在渠道创建后调用。"""
        self._sessions = sessions
        self._memory = memory
        self._todos = todos
        self._knowledge = knowledge
        self._loop_ref = loop
        self._model = model
        self._stream = stream
        self._max_input_chars = max_input_chars
        self._version = version
        self._approval_timeout = max(1.0, float(approval_timeout))
        self._policy_info = dict(policy_info or {})
        self._disclosure = dict(disclosure or {})
        self._preflight = preflight
        self._on_setup = on_setup
        self._on_test = on_test
        self._config_snapshot = config_snapshot
        self._on_config = on_config

    def update_model(self, model: str, max_input_chars: int | None = None) -> None:
        """热更新展示用的模型信息（保存首启设置后调用）。"""
        self._model = model
        if max_input_chars:
            self._max_input_chars = max_input_chars

    def update_runtime(
        self,
        *,
        stream: bool | None = None,
        max_input_chars: int | None = None,
        approval_timeout: float | None = None,
        policy_info: dict | None = None,
        disclosure: dict | None = None,
    ) -> None:
        """热更新状态接口展示的运行时信息。"""
        if stream is not None:
            self._stream = bool(stream)
        if max_input_chars:
            self._max_input_chars = max_input_chars
        if approval_timeout is not None:
            self._approval_timeout = max(1.0, float(approval_timeout))
        if policy_info is not None:
            self._policy_info = dict(policy_info)
        if disclosure is not None:
            self._disclosure = dict(disclosure)

    def config_overview(self) -> dict:
        """HTTP 线程调用：返回可编辑配置快照，绝不包含密钥。"""
        provider = self._config_snapshot
        if provider is None:
            return {"enabled": False, "config": {}, "readonly": {}}
        data = provider() or {}
        data.setdefault("enabled", True)
        return data

    def run_config_action(self, payload: dict) -> dict:
        """HTTP 线程调用：把配置修改桥接到事件循环执行。"""
        callback = self._on_config
        loop = self._loop
        if callback is None or loop is None or loop.is_closed():
            raise RuntimeError("运行时配置接口未启用")
        future = asyncio.run_coroutine_threadsafe(callback(payload), loop)
        return future.result(timeout=_SETUP_TIMEOUT)

    @property
    def setup_token(self) -> str:
        """当前进程的设置令牌；仅供 CLI 打印设置链接与测试使用。"""
        return self._setup_token

    def invalidate_setup_token(self) -> None:
        """一次性令牌：保存成功后立即失效。"""
        self._setup_token = ""

    def run_setup_action(self, kind: str, payload: dict) -> dict:
        """HTTP 线程调用：把设置/测试请求桥接到事件循环。"""
        loop = self._loop
        callback = self._on_setup if kind == "save" else self._on_test
        if callback is None or loop is None or loop.is_closed():
            raise RuntimeError("设置接口未启用")
        future = asyncio.run_coroutine_threadsafe(callback(payload), loop)
        return future.result(timeout=_SETUP_TIMEOUT)

    def _readiness(self) -> dict:
        """读取 MiniBot 的预检摘要；未注入时按"可用"处理，保持渠道可独立使用。"""
        provider = self._preflight
        if provider is None:
            return {"status": "ready", "ready": True}
        try:
            data = provider() or {}
        except Exception:  # noqa: BLE001 - 状态查询失败不能让健康检查崩掉
            logger.exception("读取就绪状态失败")
            return {"status": "blocked", "ready": False}
        data.setdefault("ready", data.get("status") == "ready")
        return data

    @property
    def ready(self) -> bool:
        return bool(self._readiness().get("ready"))

    @property
    def bound_port(self) -> int:
        """实际监听端口；测试里用 0 让系统分配时也能拿到真实值。"""
        if self._server is not None:
            return int(self._server.server_address[1])
        return self.port

    @property
    def stopping(self) -> bool:
        return self._stopping

    @property
    def max_input_chars(self) -> int:
        return self._max_input_chars

    @property
    def index_bytes(self) -> bytes:
        return self._index_bytes

    @property
    def setup_bytes(self) -> bytes:
        return self._setup_html

    async def start(self) -> None:
        if self._server is not None:
            return
        self._loop = asyncio.get_running_loop()
        try:
            with open(_INDEX_FILE, "rb") as handle:
                self._index_bytes = handle.read()
            with open(_SETUP_FILE, "rb") as handle:
                self._setup_html = handle.read()
        except OSError as exc:
            raise RuntimeError(f"Web 前端文件缺失或不可读: {exc.filename}") from exc

        self._setup_token = secrets.token_urlsafe(32)
        self._stopping = False
        server = _WebServer((self.host, self.port), _WebRequestHandler)
        server.channel = self
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="mini-agent-web",
            daemon=True,
        )
        self._thread.start()
        logger.info("Web 前端已启动: http://%s:%d", self.host, self.bound_port)

    async def stop(self) -> None:
        server = self._server
        if server is None:
            return
        self._stopping = True
        with self._subs_lock:
            streams = [item for group in self._subs.values() for item in group]
            self._subs.clear()
        for stream in streams:
            with contextlib.suppress(queue.Full):
                stream.put_nowait(None)

        self._server = None
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            await asyncio.to_thread(thread.join, 2.0)
        logger.info("Web 前端已停止")

    def set_quit_handler(self, handler) -> None:
        """注入进程级停机回调；未注入时退出接口返回 501。"""
        self._on_quit = handler if callable(handler) else None

    def request_quit(self) -> bool:
        """HTTP 线程调用：稍后触发停机，让退出响应先发出去。"""
        handler = self._on_quit
        if handler is None:
            return False
        timer = threading.Timer(0.3, handler)
        timer.daemon = True
        timer.start()
        return True

    # ------------------------------------------------------------------
    # 总线
    # ------------------------------------------------------------------
    def publish(self, session: str, text: str) -> None:
        """HTTP 线程调用：把一条消息发布进总线，最多等待 5 秒。"""
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("Web 服务尚未启动")
        future = asyncio.run_coroutine_threadsafe(
            self.bus.publish_inbound(
                InboundMessage(
                    channel=self.name,
                    sender=session,
                    text=text,
                    session_key_override=self.session_key(session),
                )
            ),
            loop,
        )
        future.result(timeout=_PUBLISH_TIMEOUT)

    async def send(self, msg: OutboundMessage) -> None:
        """事件循环调用：把增量/事件/最终回复扇出给该会话的所有 SSE 连接。"""
        if self._stopping:
            return
        if msg.event:
            event = dict(msg.event)
            event.setdefault("text", msg.text)
            event.setdefault("session", msg.to)
            if msg.side_effects:
                event["side_effects"] = msg.side_effects
            self._fanout(msg.to, event)
            return
        if msg.partial:
            event = {"type": "delta", "session": msg.to, "text": msg.text}
        else:
            event = {
                "type": "final",
                "session": msg.to,
                "text": msg.text,
                "streamed": msg.streamed,
                "citations": msg.citations or [],
                "usage": msg.usage,
                "error": msg.error,
                "retryable": msg.retryable,
                "side_effects": msg.side_effects or [],
            }
        self._fanout(msg.to, event)

    def _fanout(self, session: str, event: dict) -> None:
        with self._subs_lock:
            targets = list(self._subs.get(session, ()))
        for target in targets:
            try:
                target.put_nowait(event)
            except queue.Full:
                logger.debug("Web 会话 %s 的事件积压已满，丢弃一条事件", session)

    async def request_approval(
        self,
        *,
        session_key: str,
        sender: str,
        tool_name: str,
        arguments: dict,
    ) -> bool:
        """把副作用工具的执行请求推给浏览器，等待用户确认或超时拒绝。"""
        session = self._session_part(session_key)
        loop = self._loop or asyncio.get_running_loop()
        approval_id = secrets.token_urlsafe(12)
        future: asyncio.Future[bool] = loop.create_future()
        self._approvals[approval_id] = future
        preview = json.dumps(arguments, ensure_ascii=False)
        if len(preview) > _MAX_APPROVAL_ARGS_CHARS:
            preview = preview[:_MAX_APPROVAL_ARGS_CHARS] + "…"
        self._fanout(
            session,
            {
                "type": "approval",
                "approval_id": approval_id,
                "tool": tool_name,
                "arguments": preview,
                "session": session,
                "timeout": self._approval_timeout,
            },
        )
        try:
            return await asyncio.wait_for(future, timeout=self._approval_timeout)
        except asyncio.TimeoutError:
            logger.info("工具确认超时（%s），按拒绝处理", tool_name)
            return False
        finally:
            self._approvals.pop(approval_id, None)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """HTTP 线程调用：把用户的确认结果送回事件循环。"""
        loop = self._loop
        if loop is None or loop.is_closed():
            return False

        async def resolve() -> bool:
            future = self._approvals.get(approval_id)
            if future is None or future.done():
                return False
            future.set_result(bool(approved))
            return True

        return asyncio.run_coroutine_threadsafe(resolve(), loop).result(
            timeout=_PUBLISH_TIMEOUT
        )

    async def stop_generation(self, session: str) -> bool:
        loop_ref = self._loop_ref
        if loop_ref is None:
            return False
        return await loop_ref.cancel_turn(self.session_key(session))

    @staticmethod
    def session_key(session: str) -> str:
        return f"web:{session}"

    @staticmethod
    def _session_part(key: str) -> str:
        return key[4:] if key.startswith("web:") else key

    # ------------------------------------------------------------------
    # 只读查询（供 HTTP 线程调用）
    # ------------------------------------------------------------------
    def status(self) -> dict:
        readiness = self._readiness()
        knowledge_stats = None
        if self._knowledge is not None:
            try:
                knowledge_stats = self._knowledge.stats()
            except Exception:  # noqa: BLE001 - 状态查询失败不影响主界面
                logger.exception("读取知识库状态失败")
        return {
            "service": "mini_agent",
            "version": self._version,
            "model": self._model,
            "stream": self._stream,
            "max_input_chars": self._max_input_chars,
            "memory_enabled": bool(self._memory is not None and self._memory.enabled),
            "todos_enabled": bool(self._todos is not None and self._todos.enabled),
            "knowledge_enabled": self._knowledge is not None,
            "knowledge": knowledge_stats,
            "session_key_prefix": "web:",
            "loopback_only": True,
            "approval_timeout": self._approval_timeout,
            "policy": self._policy_info,
            "disclosure": self._disclosure,
            "can_quit": self._on_quit is not None,
            "ready": bool(readiness.get("ready")),
            "setup_required": not bool(readiness.get("ready")),
            "readiness": readiness.get("status"),
        }

    def history(self, session: str) -> list[dict[str, object]]:
        """读取会话历史；只在事件循环线程里访问 SessionManager。"""
        sessions = self._sessions
        loop = self._loop
        if sessions is None or loop is None or loop.is_closed():
            return []
        key = self.session_key(session)

        async def load() -> list[dict[str, object]]:
            record = sessions.get_or_create(key)
            messages: list[dict[str, object]] = []
            for message in record.messages:
                if message.role not in ("user", "assistant") or not message.content:
                    continue
                item: dict[str, object] = {
                    "role": message.role,
                    "content": message.content,
                }
                metadata = getattr(message, "metadata", None)
                if isinstance(metadata, dict):
                    for field in ("citations", "tools", "usage"):
                        if metadata.get(field):
                            item[field] = metadata[field]
                messages.append(item)
            return messages

        future = asyncio.run_coroutine_threadsafe(load(), loop)
        return future.result(timeout=_HISTORY_TIMEOUT)

    def _turn_running(self, session_key: str) -> bool:
        loop_ref = self._loop_ref
        checker = getattr(loop_ref, "is_turn_active", None)
        if not callable(checker):
            return False
        return bool(checker(session_key))

    def list_sessions(self) -> list[dict[str, object]]:
        """服务端会话列表；只返回 Web 前缀且至少有一条消息的会话。"""
        sessions = self._sessions
        loop = self._loop
        if sessions is None or loop is None or loop.is_closed():
            return []

        async def load() -> list[dict[str, object]]:
            items = sessions.list_sessions(prefix="web:")
            result: list[dict[str, object]] = []
            for item in items:
                if int(item.get("messages") or 0) <= 0:
                    continue
                key = str(item.get("key", ""))
                result.append(
                    {
                        "session": self._session_part(key),
                        "title": item.get("title") or "新对话",
                        "updated_at": item.get("updated_at") or "",
                        "messages": int(item.get("messages") or 0),
                        "running": self._turn_running(key),
                    }
                )
            return result

        future = asyncio.run_coroutine_threadsafe(load(), loop)
        return future.result(timeout=_HISTORY_TIMEOUT)

    async def delete_session(self, session: str) -> dict[str, int | bool]:
        """删除会话正文、长期记忆与任务清单；先停止该会话的进行中回合。"""
        key = self.session_key(session)
        stopped = await self.stop_generation(session)
        subagents = 0
        loop = self._loop_ref
        if loop is not None and getattr(loop, "subagents", None) is not None:
            subagents = await loop.subagents.cancel_session(key)
        sessions = self._sessions
        if sessions is not None:
            await asyncio.to_thread(sessions.delete, key)
        memory = 0
        if self._memory is not None:
            memory = await asyncio.to_thread(self._memory.clear, key)
        todos = 0
        if self._todos is not None:
            todos = await asyncio.to_thread(self._todos.clear, key)
        return {
            "ok": True,
            "stopped": stopped,
            "subagents_cancelled": subagents,
            "memory_removed": memory,
            "todos_removed": todos,
        }

    # ------------------------------------------------------------------
    # 知识库管理
    # ------------------------------------------------------------------
    def knowledge_overview(self) -> dict:
        kb = self._knowledge
        if kb is None:
            return {"enabled": False, "stats": None, "sources": []}
        return {
            "enabled": True,
            "stats": kb.stats(),
            "sources": kb.list_sources(),
        }

    def run_knowledge_action(self, action: str, payload: dict) -> dict:
        """HTTP 线程调用：把知识库写操作桥接到事件循环。"""
        kb = self._knowledge
        loop = self._loop
        if kb is None or loop is None or loop.is_closed():
            raise RuntimeError("知识库未启用")

        name = str(payload.get("name") or "").strip()
        content = payload.get("content")
        source = str(payload.get("source") or "").strip()

        async def run() -> dict:
            if action == "import":
                if not isinstance(content, str):
                    raise ValueError("content 必须是文本")
                count = await kb.ingest(
                    content, source=name, replace_source=True
                )
                return {
                    "ok": True,
                    "source": name,
                    "chunks": count,
                    "stats": kb.stats(),
                    "persist_error": kb.last_persist_error,
                }
            if action == "delete":
                removed = await kb.delete_source(source)
                return {
                    "ok": True,
                    "source": source,
                    "removed": removed,
                    "stats": kb.stats(),
                }
            if action == "reindex":
                count = await kb.reindex(force=True)
                return {
                    "ok": True,
                    "vectors": count,
                    "stats": kb.stats(),
                    "persist_error": kb.last_persist_error,
                }
            raise ValueError(f"未知知识库操作: {action}")

        future = asyncio.run_coroutine_threadsafe(run(), loop)
        return future.result(timeout=_KNOWLEDGE_TIMEOUT)

    def memory_entries(self, session: str) -> list[dict[str, str]] | None:
        store = self._memory
        if store is None:
            return None
        entries = store.entries(self.session_key(session))
        return [
            {"id": entry.id, "text": entry.text, "created_at": entry.created_at}
            for entry in reversed(entries[-_MAX_PANEL_ITEMS:])
        ]

    def todo_items(self, session: str) -> list[dict[str, str]] | None:
        store = self._todos
        if store is None:
            return None
        return [
            {
                "content": item.content,
                "status": item.status,
                "updated_at": item.updated_at,
            }
            for item in store.items(self.session_key(session))[-_MAX_PANEL_ITEMS:]
        ]

    # ------------------------------------------------------------------
    # SSE 订阅
    # ------------------------------------------------------------------
    def subscribe(self, session: str) -> queue.Queue | None:
        with self._subs_lock:
            total = sum(len(group) for group in self._subs.values())
            if total >= _MAX_STREAMS:
                return None
            stream: queue.Queue = queue.Queue(maxsize=_MAX_EVENT_BACKLOG)
            self._subs.setdefault(session, set()).add(stream)
        return stream

    def unsubscribe(self, session: str, stream: queue.Queue) -> None:
        with self._subs_lock:
            group = self._subs.get(session)
            if group is None:
                return
            group.discard(stream)
            if not group:
                self._subs.pop(session, None)


class _WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    channel: WebChannel


class _WebRequestHandler(BaseHTTPRequestHandler):
    server_version = "mini-agent-web"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def channel(self) -> WebChannel:
        return self.server.channel  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("web %s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if not self._guard():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                if self.channel.ready:
                    self._send_index()
                else:
                    self._send_redirect("/setup")
            elif path == "/setup":
                self._send_setup_page()
            elif path == "/healthz":
                self._send_json(200, {"ok": True, "service": "mini_agent"})
            elif path == "/readyz":
                self._send_ready()
            elif path == "/favicon.ico":
                self._send_status(204)
            elif path == "/api/status":
                self._send_json(200, self.channel.status())
            elif path == "/api/config":
                self._send_json(200, self.channel.config_overview())
            elif path == "/api/setup":
                self._send_setup_state()
            elif path == "/api/history":
                self._send_history(parsed)
            elif path == "/api/memory":
                self._send_memory(parsed)
            elif path == "/api/todos":
                self._send_todos(parsed)
            elif path == "/api/sessions":
                self._send_sessions()
            elif path == "/api/knowledge":
                self._send_json(200, self.channel.knowledge_overview())
            elif path == "/api/stream":
                self._stream(parsed)
            else:
                self._send_json(404, {"error": "接口不存在"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:  # noqa: BLE001 - 单个请求失败不能拖垮服务
            logger.exception("Web 请求处理失败: %s", path)
            with contextlib.suppress(Exception):
                self._send_json(500, {"error": "服务器内部错误"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        try:
            if not self._guard():
                return
            path = urlparse(self.path).path
            try:
                if path == "/api/message":
                    self._handle_message()
                elif path == "/api/stop":
                    self._handle_stop()
                elif path == "/api/quit":
                    self._handle_quit()
                elif path == "/api/approval":
                    self._handle_approval()
                elif path == "/api/session/delete":
                    self._handle_session_delete()
                elif path == "/api/knowledge/import":
                    self._handle_knowledge_import()
                elif path == "/api/knowledge/delete":
                    self._handle_knowledge_delete()
                elif path == "/api/knowledge/reindex":
                    self._handle_knowledge_reindex()
                elif path == "/api/config":
                    self._handle_config_post()
                elif path == "/api/setup":
                    self._handle_setup_post("save")
                elif path == "/api/setup/test":
                    self._handle_setup_post("test")
                else:
                    self._send_json(404, {"error": "接口不存在"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:  # noqa: BLE001 - 单个请求失败不能拖垮服务
                logger.exception("Web 请求处理失败: %s", path)
                with contextlib.suppress(Exception):
                    self._send_json(500, {"error": "服务器内部错误"})
        finally:
            # 被拒绝的请求（令牌/来源校验失败）没有解析请求体；不丢弃会让
            # keep-alive 连接里的残余字节污染下一个请求，Windows 上还会 RST。
            self._drain_body()

    def _handle_message(self) -> None:
        if not self.channel.ready:
            self._send_json(
                409, {"error": "尚未完成配置，请先在设置页完成配置", "setup": "/setup"}
            )
            return
        payload = self._read_json()
        if payload is None:
            return
        session = payload.get("session")
        text = payload.get("text")
        if not isinstance(session, str) or not _SESSION_RE.match(session):
            self._send_json(
                400, {"error": "session 必须是 1-64 位字母、数字、下划线或连字符"}
            )
            return
        if not isinstance(text, str) or not text.strip():
            self._send_json(400, {"error": "消息不能为空"})
            return
        if len(text) > self.channel.max_input_chars:
            self._send_json(
                413,
                {
                    "error": f"消息过长：{len(text)} 字符，"
                    f"上限 {self.channel.max_input_chars} 字符"
                },
            )
            return
        try:
            self.channel.publish(session, text)
        except TimeoutError:
            self._send_json(429, {"error": "服务繁忙，请稍后重试"})
            return
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
        self._send_json(202, {"ok": True, "session": session})

    def _handle_stop(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        session = payload.get("session")
        if not isinstance(session, str) or not _SESSION_RE.match(session):
            self._send_json(400, {"error": "session 不合法"})
            return
        try:
            stopped = self._run_async(self.channel.stop_generation(session), 10.0)
        except TimeoutError:
            self._send_json(504, {"error": "停止生成超时，请稍后重试"})
            return
        self._send_json(200, {"ok": True, "stopped": stopped})

    def _handle_quit(self) -> None:
        """界面退出：只接受显式确认，没有 CLI 停机回调时如实拒绝。"""
        payload = self._read_json()
        if payload is None:
            return
        if payload.get("confirm") is not True:
            self._send_json(400, {"error": "退出需要显式确认"})
            return
        if not self.channel.request_quit():
            self._send_json(
                501, {"error": "当前启动方式不支持从界面退出，请回到终端按 Ctrl+C"}
            )
            return
        self._send_json(200, {"ok": True, "message": "服务正在退出"})

    def _handle_approval(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        approval_id = payload.get("approval_id")
        approved = payload.get("approved")
        if (
            not isinstance(approval_id, str)
            or not _SESSION_RE.match(approval_id)
            or not isinstance(approved, bool)
        ):
            self._send_json(400, {"error": "approval_id 或 approved 不合法"})
            return
        try:
            resolved = self.channel.resolve_approval(approval_id, approved)
        except TimeoutError:
            self._send_json(504, {"error": "确认结果提交超时"})
            return
        self._send_json(200, {"ok": resolved})

    def _handle_session_delete(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        session = payload.get("session")
        if not isinstance(session, str) or not _SESSION_RE.match(session):
            self._send_json(400, {"error": "session 不合法"})
            return
        try:
            result = self._run_async(self.channel.delete_session(session), 20.0)
        except TimeoutError:
            self._send_json(504, {"error": "删除会话超时"})
            return
        self._send_json(200, result)

    def _handle_knowledge_import(self) -> None:
        payload = self._read_json(limit=_MAX_UPLOAD_BYTES)
        if payload is None:
            return
        name = payload.get("name")
        content = payload.get("content")
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 200
            or os.path.basename(name) != name
            or "\x00" in name
            or name in (".", "..")
        ):
            self._send_json(400, {"error": "文档名必须是 1-200 字符的纯文件名"})
            return
        if not isinstance(content, str) or not content.strip():
            self._send_json(400, {"error": "文档内容不能为空"})
            return
        if len(content) > _MAX_UPLOAD_CHARS:
            self._send_json(
                413,
                {"error": f"文档过长：上限 {_MAX_UPLOAD_CHARS} 字符"},
            )
            return
        if "\x00" in content[:4096]:
            self._send_json(400, {"error": "文档看起来是二进制内容"})
            return
        try:
            result = self.channel.run_knowledge_action(
                "import", {"name": name.strip(), "content": content}
            )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
        except TimeoutError:
            self._send_json(504, {"error": "文档摄入超时，请稍后重试"})
            return
        self._send_json(200, result)

    def _handle_knowledge_delete(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        source = payload.get("source")
        if not isinstance(source, str) or not source.strip() or len(source) > 200:
            self._send_json(400, {"error": "source 必须是 1-200 个字符"})
            return
        try:
            result = self.channel.run_knowledge_action(
                "delete", {"source": source.strip()}
            )
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
        except TimeoutError:
            self._send_json(504, {"error": "删除文档超时"})
            return
        self._send_json(200, result)

    def _handle_knowledge_reindex(self) -> None:
        payload = self._read_json(allow_empty=True)
        if payload is None:
            return
        try:
            result = self.channel.run_knowledge_action("reindex", {})
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
        except TimeoutError:
            self._send_json(504, {"error": "重建索引超时"})
            return
        self._send_json(200, result)

    def _run_async(self, coro, timeout: float):
        loop = self.channel._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("服务尚未启动")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    def _handle_config_post(self) -> None:
        """校验并热应用运行时配置；密钥不在本接口的允许字段内。"""
        payload = self._read_json()
        if payload is None:
            return
        try:
            result = self.channel.run_config_action(payload)
        except ConfigError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
        except TimeoutError:
            self._send_json(504, {"error": "应用配置超时，请稍后重试"})
            return
        except Exception:  # noqa: BLE001 - 单个请求失败不能拖垮服务
            logger.exception("应用运行时配置失败")
            self._send_json(500, {"error": "服务器内部错误"})
            return
        self._send_json(200, result)

    def _handle_setup_post(self, kind: str) -> None:
        """保存设置 / 测试连接；令牌校验先于读取请求体。"""
        readiness = self.channel._readiness()
        if kind == "save" and readiness.get("ready"):
            self._send_json(409, {"error": "配置已完成，设置接口已关闭"})
            return
        expected = self.channel.setup_token
        if not expected:
            self._send_json(
                503, {"error": "设置接口未启用，请重启后使用终端打印的设置链接"}
            )
            return
        provided = self.headers.get("X-Setup-Token") or ""
        if not secrets.compare_digest(provided, expected):
            self._send_json(
                403, {"error": "设置令牌无效或已失效，请使用启动时打印的设置链接"}
            )
            return

        payload = self._read_json()
        if payload is None:
            return
        model = payload.get("model")
        api_base = payload.get("api_base")
        api_key = payload.get("api_key")
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            self._send_json(400, {"error": "模型名必须是 1-200 个字符"})
            return
        if (
            not isinstance(api_base, str)
            or not api_base.strip()
            or len(api_base) > 2048
        ):
            self._send_json(400, {"error": "端点地址必须是 1-2048 个字符的 http(s) 地址"})
            return
        if api_key is not None and (not isinstance(api_key, str) or len(api_key) > 8192):
            self._send_json(400, {"error": "API 密钥格式不合法"})
            return

        request = {
            "model": model.strip(),
            "api_base": api_base.strip(),
            "api_key": api_key,
        }
        try:
            result = self.channel.run_setup_action(kind, request)
        except ConfigError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except credentials.CredentialsError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except TimeoutError:
            self._send_json(504, {"error": "操作超时，请重试"})
            return
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
        except Exception:  # noqa: BLE001
            logger.exception("设置请求失败")
            self._send_json(500, {"error": "服务器内部错误"})
            return
        if kind == "save" and result.get("ok"):
            self.channel.invalidate_setup_token()
        self._send_json(200, result)

    # ------------------------------------------------------------------
    # 请求校验
    # ------------------------------------------------------------------
    def _guard(self) -> bool:
        """只接受本机来源；Host / Origin 可防 DNS rebinding 与跨站请求。"""
        remote = self.client_address[0]
        if remote not in _LOOPBACK_HOSTS:
            self._send_json(403, {"error": "只允许本机访问"})
            return False
        host = self._header_host()
        if host and host not in _LOOPBACK_HOSTS:
            self._send_json(403, {"error": "Host 不在允许范围内"})
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
                self._send_json(403, {"error": "Origin 不在允许范围内"})
                return False
        return True

    def _header_host(self) -> str:
        raw = self.headers.get("Host") or ""
        parsed = urlparse(f"//{raw}")
        return (parsed.hostname or "").lower()

    def _session_param(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("session") or []
        session = values[0] if values else ""
        if not _SESSION_RE.match(session):
            self._send_json(
                400, {"error": "session 必须是 1-64 位字母、数字、下划线或连字符"}
            )
            return None
        return session

    def _discard_body(self, length: int) -> bool:
        """读取并丢弃 length 字节；超过可安全丢弃上限时返回 False。"""
        if length > _MAX_DRAIN_BYTES:
            return False
        remaining = max(length, 0)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                return False
            remaining -= len(chunk)
        return True

    def _drain_body(self) -> None:
        """丢弃未被解析的请求体；幂等，响应前后都可安全调用。"""
        if getattr(self, "_body_consumed", False):
            return
        self._body_consumed = True
        raw_length = self.headers.get("Content-Length")
        try:
            remaining = int(raw_length or 0)
        except ValueError:
            return
        if not self._discard_body(remaining):
            # 超大请求体读不完，残余字节无法同步，直接断开连接。
            self.close_connection = True

    def _read_json(
        self, *, limit: int = _MAX_BODY_BYTES, allow_empty: bool = False
    ) -> dict | None:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or 0)
        except ValueError:
            self._send_json(400, {"error": "Content-Length 不合法"})
            return None
        if length <= 0:
            if allow_empty:
                self._body_consumed = True
                return {}
            self._send_json(400, {"error": "请求体为空"})
            return None
        if length > limit:
            # 先尽量把请求体读掉再回 413，否则 Windows 会在关闭时发 RST，
            # 客户端还没读到响应就报连接中断。
            self._body_consumed = True
            if not self._discard_body(length):
                self.close_connection = True
            self._send_json(413, {"error": "请求体过大"})
            return None
        self._body_consumed = True
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "请求体必须是 UTF-8 JSON"})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "请求体必须是 JSON 对象"})
            return None
        return payload

    # ------------------------------------------------------------------
    # 响应
    # ------------------------------------------------------------------
    def _send_index(self) -> None:
        body = self.channel.index_bytes
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_setup_page(self) -> None:
        body = self.channel.setup_bytes
        if not body:
            self._send_json(503, {"error": "设置页未加载"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_ready(self) -> None:
        data = self.channel._readiness()
        ok = bool(data.get("ready"))
        self._send_json(200 if ok else 503, {"ok": ok, **data})

    def _send_setup_state(self) -> None:
        data = self.channel._readiness()
        payload = dict(data)
        payload.update(
            {
                "setup_required": not bool(data.get("ready")),
                "setup_token_required": bool(self.channel.setup_token),
                "providers": _PROVIDER_TEMPLATES,
                "keyring_available": credentials.keyring_available(),
            }
        )
        self._send_json(200, payload)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if status != 204:
            self.wfile.write(body)

    def _send_status(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_sessions(self) -> None:
        try:
            items = self.channel.list_sessions()
        except TimeoutError:
            self._send_json(503, {"error": "读取会话列表超时，请稍后重试"})
            return
        self._send_json(200, {"sessions": items})

    def _send_history(self, parsed) -> None:
        session = self._session_param(parsed)
        if session is None:
            return
        try:
            messages = self.channel.history(session)
        except TimeoutError:
            self._send_json(503, {"error": "读取会话历史超时，请稍后重试"})
            return
        self._send_json(200, {"session": session, "messages": messages})

    def _send_memory(self, parsed) -> None:
        session = self._session_param(parsed)
        if session is None:
            return
        entries = self.channel.memory_entries(session)
        self._send_json(
            200,
            {
                "session": session,
                "enabled": entries is not None,
                "entries": entries or [],
            },
        )

    def _send_todos(self, parsed) -> None:
        session = self._session_param(parsed)
        if session is None:
            return
        items = self.channel.todo_items(session)
        self._send_json(
            200,
            {
                "session": session,
                "enabled": items is not None,
                "items": items or [],
            },
        )

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------
    def _stream(self, parsed) -> None:
        session = self._session_param(parsed)
        if session is None:
            return
        stream = self.channel.subscribe(session)
        if stream is None:
            self._send_json(429, {"error": "同时打开的前端窗口过多"})
            return

        # 没有 Content-Length，用 Connection: close 界定响应体，浏览器按到达即读。
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n")
            self._write_event({"type": "ready", "session": session})
            while not self.channel.stopping:
                try:
                    event = stream.get(timeout=_SSE_PING_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    continue
                if event is None:
                    break
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.channel.unsubscribe(session, stream)

    def _write_event(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
