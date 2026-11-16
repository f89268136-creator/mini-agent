"""MiniBot：对外唯一门面，负责把各部件装配成可运行的系统。

    MiniBot(config)
        ├── MessageBus
        ├── SessionManager
        ├── AgentRunner(build_provider(config))
        └── AgentLoop(bus, runner, sessions, channels)

构造是同步的：不建立网络连接、不起后台任务；``start()`` 才启动分发器与渠道。
``chat()`` 可以脱离 ``start()`` 使用（脚本/测试/嵌入式场景），仍然走会话锁、
历史持久化与完整 ReAct 流程。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import replace
from pathlib import Path

from .agent.commands import CommandHandler
from .agent.context import ContextBuilder
from .agent.skills import SkillRegistry
from .agent.loop import AgentLoop
from .agent.runner import AgentRunner
from .agent.subagent import SubagentManager
from .agent.token_budget import estimate_tool_schemas_tokens
from .agent.tools import (
    all_tool_schemas,
    configure_knowledge_base,
    configure_policy,
    configure_tool_timeout,
    describe_policy,
)
from .agent.tools.knowledge import get_knowledge_base
from . import __version__
from .bus import MessageBus
from .channels import Channel, WebChannel, build_channels, validate_channels
from .config import (
    ConfigError,
    MiniConfig,
    MissingCredentialsError,
    load_config,
    load_config_report,
    load_file_config,
    write_config,
)
from .memory import build_memory_backend, configure_memory_store
from .providers import build_embedder, build_provider
from .providers.base import DeltaCallback
from .runtime.preflight import PreflightResult, run_preflight
from .session import SessionManager
from .session.todo import TodoStore, configure_todo_store
from .utils import credentials
from .utils.tracing import configure_tracing, redact_trace_payload, shutdown_tracing

logger = logging.getLogger(__name__)

# 工具策略、长期记忆仓库、任务清单、RAG、追踪与子代理管理器都是进程级单例。
# 记录第一次装配时的指纹，检测到同进程用不同配置再建实例时给出警告。
_PROCESS_GLOBAL_FIELDS = (
    "workspace",
    "allow_shell",
    "allow_private_urls",
    "memory_enabled",
    "memory_backend",
    "memory_dir",
    "mem0_dir",
    "todos_enabled",
    "todos_dir",
    "todo_max_items",
    "langfuse_enabled",
    "langfuse_base_url",
    "langfuse_sample_rate",
    "langfuse_redact",
    "embedding_model",
    "embedding_backend",
    "knowledge_chunk_size",
    "knowledge_dir",
    "knowledge_chunk_overlap",
    "knowledge_top_k",
    "knowledge_min_score",
    "knowledge_lexical_weight",
    "knowledge_mmr_lambda",
    "multi_agent_enabled",
    "max_concurrent_subagents",
    "subagent_timeout",
    "tool_timeout",
)
_configured_fingerprint: tuple[str, ...] | None = None
_fingerprint_lock = threading.Lock()

# Web 设置面板允许热更新的字段；其余字段保存后需要重启才能生效。
_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "model",
        "stream",
        "max_input_chars",
        "max_iterations",
        "turn_timeout",
        "tool_timeout",
        "tool_approval",
        "approval_timeout",
        "max_concurrent_turns",
        "max_active_sessions",
        "session_idle_timeout",
        "history_limit",
        "prompt_max_chars",
        "prompt_max_tokens",
        "context_compact_ratio",
        "context_keep_turns",
        "context_summary_max_chars",
        "memory_inject_chars",
        "todo_inject_chars",
        "memory_enabled",
        "memory_max_entries",
        "todos_enabled",
        "todo_max_items",
        "allow_shell",
        "allow_private_urls",
        "knowledge_top_k",
        "knowledge_min_score",
        "knowledge_lexical_weight",
        "knowledge_mmr_lambda",
        "multi_agent_enabled",
        "max_concurrent_subagents",
        "subagent_timeout",
        "subagent_max_result_chars",
        "retry_max_attempts",
        "retry_base_delay",
        "timeout",
    }
)
_PROVIDER_CONFIG_FIELDS = frozenset(
    {"model", "retry_max_attempts", "retry_base_delay", "timeout"}
)
_CONTEXT_CONFIG_FIELDS = frozenset(
    {
        "prompt_max_chars",
        "prompt_max_tokens",
        "memory_inject_chars",
        "todo_inject_chars",
        # 这些开关会改变发给模型的工具 Schema，进而影响输入 token 预算。
        "allow_shell",
        "memory_enabled",
        "todos_enabled",
        "multi_agent_enabled",
    }
)
_SUBAGENT_CONFIG_FIELDS = frozenset(
    {
        "multi_agent_enabled",
        "max_concurrent_subagents",
        "subagent_timeout",
        "subagent_max_result_chars",
    }
)
# 只读展示：修改这些项必须编辑配置文件并重启进程。
_READONLY_CONFIG_FIELDS = (
    "api_base",
    "credential_source",
    "workspace",
    "channels",
    "echo_port",
    "web_port",
    "max_queue_size",
    "log_level",
    "sessions_dir",
    "memory_backend",
    "memory_dir",
    "todos_dir",
    "knowledge_dir",
    "knowledge_chunk_size",
    "knowledge_chunk_overlap",
    "embedding_backend",
    "embedding_model",
    "langfuse_enabled",
)


def _check_process_globals(config: MiniConfig) -> None:
    global _configured_fingerprint
    fingerprint = tuple(repr(getattr(config, name)) for name in _PROCESS_GLOBAL_FIELDS)
    with _fingerprint_lock:
        previous = _configured_fingerprint
        _configured_fingerprint = fingerprint
    if previous is not None and previous != fingerprint:
        logger.warning(
            "同一进程内用不同配置创建了多个 MiniBot：工具策略、记忆、任务清单、"
            "追踪等是进程级资源，后创建的实例会覆盖先前的设置。"
            "多租户或多套配置请为每个实例使用独立进程。"
        )


def _as_optional_str(value) -> str | None:
    return value if isinstance(value, str) else None


def _validate_api_key(value: str | None) -> str | None:
    """拒绝会破坏 HTTP 头的控制字符；空字符串视为未填写。"""
    if not value:
        return None
    key = value.strip()
    if not key:
        return None
    if len(key) > 8192 or any(ch in key for ch in "\r\n\0"):
        raise ConfigError("API 密钥格式不合法（包含换行/控制字符或长度超限）")
    return key


async def _close_quietly(resource) -> None:
    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 - 关闭旧资源失败不应影响新配置
        logger.exception("关闭被替换的资源失败")


async def _close_after_turns(resource, loop: AgentLoop, timeout: float = 900.0) -> None:
    """等进行中的回合结束后再关闭旧模型客户端，避免热更新打断正在生成的回复。"""
    deadline = asyncio.get_running_loop().time() + timeout
    while loop.stats().get("active_turns", 0):
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning("等待旧模型客户端空闲超时，仍执行关闭")
            break
        await asyncio.sleep(0.5)
    await _close_quietly(resource)


class MiniBot:
    def __init__(
        self,
        config: MiniConfig | None = None,
        *,
        allow_setup: bool = False,
        preflight: PreflightResult | None = None,
        config_path: str | None = None,
    ):
        """``allow_setup=True`` 供首启设置页使用：缺密钥也能把 Web 服务跑起来。

        普通 SDK 用法保持构造即校验凭据，避免直到第一次请求才暴露 401。
        """
        self.config: MiniConfig = config or load_config()
        if not allow_setup:
            self.config.validate_credentials()
        self.config_path: str | None = config_path
        self.preflight: PreflightResult | None = preflight
        validate_channels(self.config.channels)
        self._started = False
        self._closed = False
        _check_process_globals(self.config)

        configure_tracing(
            enabled=self.config.langfuse_enabled,
            base_url=self.config.langfuse_base_url,
            sample_rate=self.config.langfuse_sample_rate,
            mask=redact_trace_payload if self.config.langfuse_redact else None,
        )

        # 工具策略与知识库是进程级资源；一个进程应只运行一套配置。
        configure_policy(
            workspace=self.config.workspace,
            allow_shell=self.config.allow_shell,
            allow_private_urls=self.config.allow_private_urls,
            tool_approval=self.config.tool_approval,
        )
        configure_tool_timeout(self.config.tool_timeout)
        self._configure_knowledge(self.config)

        self.bus = MessageBus(max_queue_size=self.config.max_queue_size)
        self.sessions = SessionManager(
            self.config.sessions_dir,
            self.config.system_prompt,
            self.config.history_limit,
        )
        self.memory = build_memory_backend(self.config)
        configure_memory_store(self.memory)
        self.todo = TodoStore(
            self.config.todos_dir,
            max_items=self.config.todo_max_items,
            enabled=self.config.todos_enabled,
        )
        configure_todo_store(self.todo)
        self.skills = SkillRegistry(
            self._skill_roots(self.config),
            enabled=self.config.skills_enabled,
        )
        self.skills.scan()
        self.context = self._build_context(self.config)
        self.runner = AgentRunner(
            build_provider(self.config),
            self.config.max_iterations,
        )
        self.subagents = self._build_subagents(self.config, self.runner.provider)
        self.channels: dict[str, Channel] = build_channels(
            self.config.channels,
            self.bus,
            echo_port=self.config.echo_port,
            web_port=self.config.web_port,
        )
        self.loop = AgentLoop(
            self.bus,
            self.runner,
            self.sessions,
            channels=self.channels,
            session_idle_timeout=self.config.session_idle_timeout,
            max_input_chars=self.config.max_input_chars,
            max_active_sessions=self.config.max_active_sessions,
            session_queue_size=self.config.max_queue_size,
            turn_timeout=self.config.turn_timeout,
            max_concurrent_turns=self.config.max_concurrent_turns,
            context=self.context,
            subagents=self.subagents,
            stream=self.config.stream,
            prompt_max_tokens=self.config.prompt_max_tokens,
            context_compact_ratio=self.config.context_compact_ratio,
            context_keep_turns=self.config.context_keep_turns,
            context_summary_max_chars=self.config.context_summary_max_chars,
            skills=self.skills,
        )
        web_channel = self.channels.get("web")
        if isinstance(web_channel, WebChannel):
            # Web 渠道只读地展示会话/记忆/清单/文档；写入仍统一走 AgentLoop。
            # loop 注入后，停止生成与删除会话才能取消进行中的回合与子代理。
            web_channel.configure(
                sessions=self.sessions,
                memory=self.memory,
                todos=self.todo,
                knowledge=get_knowledge_base(),
                loop=self.loop,
                model=self.config.model,
                stream=self.config.stream,
                max_input_chars=self.config.max_input_chars,
                version=__version__,
                approval_timeout=self.config.approval_timeout,
                policy_info=describe_policy(),
                disclosure=self._disclosure(),
                preflight=self.readiness,
                on_setup=self._on_setup_request,
                on_test=self._on_setup_test,
                config_snapshot=self.runtime_config_snapshot,
                on_config=self.apply_runtime_settings,
            )

    def _build_context(self, config: MiniConfig) -> ContextBuilder:
        return ContextBuilder(
            config.system_prompt,
            self.memory,
            max_chars=config.prompt_max_chars,
            max_tokens=config.prompt_max_tokens,
            memory_inject_chars=config.memory_inject_chars,
            todo=self.todo,
            todo_inject_chars=config.todo_inject_chars,
            workspace=config.workspace,
            skills=self.skills,
            tool_tokens=estimate_tool_schemas_tokens(all_tool_schemas()),
        )

    def _disclosure(self) -> dict:
        """说明哪些数据会离开本机；展示在 Web 状态接口里。"""
        return {
            "model_endpoint": self.config.api_base,
            "sends_to_model": [
                "对话历史与压缩摘要",
                "长期记忆摘要",
                "任务清单",
                "技能目录与调用时的技能正文",
                "工具参数与结果",
                "检索到的文档片段",
            ],
            "tracing": {
                "enabled": self.config.langfuse_enabled,
                "redacted": self.config.langfuse_redact,
            },
            "local_only": [
                "会话正文",
                "文档索引",
                "长期记忆",
                "任务清单",
                "技能包文件",
            ],
        }

    @staticmethod
    def _skill_roots(config: MiniConfig) -> list[tuple[str, Path]]:
        workspace = Path(config.workspace).expanduser().resolve()
        return [
            ("user", Path.home() / ".mini_agent" / "skills"),
            ("project", workspace / ".mini_agent_skills"),
            ("project", workspace / ".mini_agent" / "skills"),
        ]

    def _configure_knowledge(self, config: MiniConfig) -> None:
        """按配置重建进程级知识库；索引目录让片段与向量跨进程保留。"""
        self._embedder = build_embedder(config)
        configure_knowledge_base(
            self._embedder,
            chunk_size=config.knowledge_chunk_size,
            overlap=config.knowledge_chunk_overlap,
            index_dir=config.knowledge_dir,
            min_score=config.knowledge_min_score,
            lexical_weight=config.knowledge_lexical_weight,
            mmr_lambda=config.knowledge_mmr_lambda,
            default_top_k=config.knowledge_top_k,
        )

    # ------------------------------------------------------------------
    # 运行时配置：Web 设置面板（端点与密钥不在此接口内）
    # ------------------------------------------------------------------
    def runtime_config_snapshot(self) -> dict:
        """返回设置面板需要的当前值；密钥永不返回。"""
        readonly = {
            name: getattr(self.config, name) for name in _READONLY_CONFIG_FIELDS
        }
        readonly["channels"] = list(self.config.channels)
        return {
            "enabled": True,
            "config": {
                name: getattr(self.config, name)
                for name in sorted(_RUNTIME_CONFIG_FIELDS)
            },
            "readonly": readonly,
        }

    async def apply_runtime_settings(self, payload: dict) -> dict:
        """校验 -> 持久化 -> 热应用；任何校验失败都不会改动运行中的配置。"""
        if not isinstance(payload, dict):
            raise ConfigError("配置必须是 JSON 对象")
        if "api_key" in payload or "api_base" in payload:
            raise ConfigError("端点与密钥请使用首启设置页或 mini-agent login 修改")
        unknown = sorted(set(payload) - _RUNTIME_CONFIG_FIELDS)
        if unknown:
            raise ConfigError("不支持在运行时修改：" + "、".join(unknown))
        values = dict(payload)
        candidate = replace(self.config, **values)
        candidate.validate()

        # 写回基准只取文件层：CLI 的端口/渠道等运行时覆盖不应被固化。
        file_candidate = replace(load_file_config(self.config_path), **values)
        file_candidate.validate()
        try:
            config_path = write_config(file_candidate, self.config_path)
        except OSError as exc:
            raise ConfigError(f"无法写入配置文件: {exc.strerror or exc}") from exc

        rebuild_provider = bool(_PROVIDER_CONFIG_FIELDS & values.keys())
        rebuild_subagents = rebuild_provider or bool(
            (_SUBAGENT_CONFIG_FIELDS | {"max_iterations"}) & values.keys()
        )
        new_provider = build_provider(candidate) if rebuild_provider else None
        new_runner = (
            AgentRunner(new_provider, candidate.max_iterations)
            if new_provider is not None
            else None
        )
        new_subagents = (
            self._build_subagents(candidate, new_provider or self.runner.provider)
            if rebuild_subagents
            else None
        )
        old_provider = self.runner.provider if new_provider is not None else None
        old_subagents = self.subagents if rebuild_subagents else None

        self.config = candidate
        self.sessions.history_limit = candidate.history_limit
        if self.memory is not None:
            self.memory.enabled = candidate.memory_enabled
            if hasattr(self.memory, "max_entries"):
                self.memory.max_entries = candidate.memory_max_entries
        if self.todo is not None:
            self.todo.enabled = candidate.todos_enabled
            self.todo.max_items = candidate.todo_max_items
        knowledge = get_knowledge_base()
        if knowledge is not None:
            knowledge.min_score = candidate.knowledge_min_score
            knowledge.lexical_weight = candidate.knowledge_lexical_weight
            knowledge.mmr_lambda = candidate.knowledge_mmr_lambda
            knowledge.default_top_k = candidate.knowledge_top_k
        configure_policy(
            workspace=candidate.workspace,
            allow_shell=candidate.allow_shell,
            allow_private_urls=candidate.allow_private_urls,
            tool_approval=candidate.tool_approval,
        )
        configure_tool_timeout(candidate.tool_timeout)
        self.loop.configure_limits(
            max_input_chars=candidate.max_input_chars,
            turn_timeout=candidate.turn_timeout,
            prompt_max_tokens=candidate.prompt_max_tokens,
            context_compact_ratio=candidate.context_compact_ratio,
            context_keep_turns=candidate.context_keep_turns,
            context_summary_max_chars=candidate.context_summary_max_chars,
            session_idle_timeout=candidate.session_idle_timeout,
            max_active_sessions=candidate.max_active_sessions,
            max_concurrent_turns=candidate.max_concurrent_turns,
            stream=candidate.stream,
        )
        if new_runner is not None:
            self.runner = new_runner
        else:
            self.runner.max_iterations = candidate.max_iterations
        if rebuild_subagents:
            self.subagents = new_subagents
        self.loop.rebind(runner=self.runner, subagents=self.subagents)
        # 必须放在 rebind 之后：子代理工具是否可见由 rebind 更新；
        # 工具 Schema 集合决定 token 预算，提前构建会写入旧的 Schema 开销。
        if _CONTEXT_CONFIG_FIELDS & values.keys():
            self.loop.context = self._build_context(candidate)

        web_channel = self.channels.get("web")
        if isinstance(web_channel, WebChannel):
            web_channel.update_model(candidate.model, candidate.max_input_chars)
            web_channel.update_runtime(
                stream=candidate.stream,
                max_input_chars=candidate.max_input_chars,
                approval_timeout=candidate.approval_timeout,
                policy_info=describe_policy(),
                disclosure=self._disclosure(),
            )

        # 进行中的回合仍持有旧 provider：有活动回合时延后关闭，避免热更新
        # 把正在生成的回复打断。
        replacements = [
            item
            for item in (old_subagents, old_provider)
            if item is not None
            and item is not self.subagents
            and item is not self.runner.provider
        ]
        if replacements:
            if self.loop.stats().get("active_turns", 0):
                for item in replacements:
                    asyncio.create_task(_close_after_turns(item, self.loop))
            else:
                for item in replacements:
                    await _close_quietly(item)

        logger.info("运行时配置已更新: %s", ", ".join(sorted(values)))
        return {
            "ok": True,
            "config_path": config_path,
            "applied": sorted(values),
            "config": self.runtime_config_snapshot()["config"],
        }

    @classmethod
    def from_config(
        cls, path: str | None = None, *, allow_setup: bool = False
    ) -> "MiniBot":
        """从配置文件（缺省时按环境变量与默认路径查找）构造。"""
        bot = cls(
            load_config(path, require_credentials=not allow_setup),
            allow_setup=allow_setup,
        )
        bot.config_path = path
        return bot

    async def chat(
        self,
        text: str,
        session_key: str = "sdk:local",
        on_delta: DeltaCallback | None = None,
    ) -> str:
        """直接问一句并返回回答；历史按 session_key 正常存取与持久化。

        同一 session_key 的并发调用会被 AgentLoop 的会话锁串行化，
        不会出现两条消息交叉写入历史的情况。

        ``on_delta`` 用于接收流式文本增量；配置关闭 stream 时会被忽略，
        仍然返回完整回答。
        """
        if self._closed:
            raise RuntimeError("MiniBot 已关闭，不能继续调用 chat()")
        outbound = await self.loop.handle_text(
            channel="sdk",
            sender="local",
            text=text,
            session_key=session_key,
            on_delta=on_delta if self.config.stream else None,
        )
        return outbound.text

    def register_command(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
        usage: str = "",
        aliases: tuple[str, ...] = (),
    ) -> None:
        """登记一个斜杠命令；handler 接收 CommandContext 并返回要展示的文本。"""
        self.loop.commands.register(
            name,
            handler,
            description=description,
            usage=usage,
            aliases=aliases,
        )

    def reload_skills(self) -> int:
        """重新扫描用户级/项目级技能目录，返回技能数量。"""
        return self.skills.scan()

    # ------------------------------------------------------------------
    # 首启设置：保存凭据 + 热重建模型组件
    # ------------------------------------------------------------------
    def readiness(self) -> dict:
        """给 /api/status 与 /readyz 用的只读状态，不含密钥。"""
        if self.preflight is not None:
            return self.preflight.as_dict()
        status = "setup_required" if self.config.needs_setup() else "ready"
        return {
            "status": status,
            "ready": status == "ready",
            "config": {
                "model": self.config.model,
                "api_base": self.config.api_base,
                "credential_source": self.config.credential_source,
                "needs_setup": self.config.needs_setup(),
            },
        }

    async def apply_setup(
        self, *, model: str, api_base: str, api_key: str | None = None
    ) -> dict:
        """保存设置并热重建 provider，调用方无需重启进程。

        密钥只进系统钥匙串（或显式凭据文件），配置文件只写与默认值的差异。
        """
        model = (model or "").strip()
        api_base = (api_base or "").strip()
        if not model or not api_base:
            raise ConfigError("模型名与端点不能为空")
        api_key = _validate_api_key(api_key)

        # 写回基准只取文件层：CLI 的 --port / --model 等运行时覆盖不应被固化。
        # 换服务商时不能沿用旧端点的密钥，所以这里从空密钥开始重新解析。
        candidate = replace(
            load_file_config(self.config_path),
            model=model,
            api_base=api_base,
            api_key=None,
        )
        source = ""
        if api_key:
            source = credentials.save_api_key(candidate.api_base, api_key)
            candidate.api_key = api_key
        else:
            key, source = credentials.load_api_key(candidate.api_base)
            candidate.api_key = key
        candidate.credential_source = source
        if candidate.needs_setup():
            raise MissingCredentialsError(
                "该端点需要 API 密钥；请填写后重试（密钥会存入系统钥匙串）"
            )

        try:
            config_path = write_config(candidate, self.config_path)
        except OSError as exc:
            raise ConfigError(f"无法写入配置文件: {exc.strerror or exc}") from exc

        report = load_config_report(self.config_path)
        if report.needs_repair:
            raise ConfigError("；".join(report.errors))
        effective = report.config
        if effective.needs_setup():
            raise MissingCredentialsError(
                "凭据保存后仍无法读取；请改运行 mini-agent login 完成配置"
            )

        new_provider = build_provider(effective)
        new_runner = AgentRunner(new_provider, effective.max_iterations)
        new_subagents = self._build_subagents(effective, new_provider)
        old_provider = self.runner.provider
        old_subagents = self.subagents

        # 只把设置页负责的字段换到运行配置上，保留渠道/端口等本次启动的覆盖。
        self.config = replace(
            self.config,
            model=effective.model,
            api_base=effective.api_base,
            api_key=effective.api_key,
        )
        self.config.credential_source = effective.credential_source
        self.runner = new_runner
        self.subagents = new_subagents
        self.loop.rebind(runner=new_runner, subagents=new_subagents)
        # 嵌入客户端持有旧密钥/端点，必须一起替换；索引会按新指纹自动重建向量。
        old_embedder = self._embedder
        self._configure_knowledge(self.config)
        web_channel = self.channels.get("web")
        if isinstance(web_channel, WebChannel):
            web_channel.update_model(effective.model, effective.max_input_chars)
        self.preflight = run_preflight(report)

        await _close_quietly(old_provider)
        await _close_quietly(old_embedder)
        if old_subagents is not None:
            await _close_quietly(old_subagents)
        logger.info("首启设置已保存: model=%s endpoint=%s", effective.model, effective.api_base)
        return {
            "ok": True,
            "config_path": config_path,
            "credential_source": effective.credential_source,
            "preflight": self.preflight.as_dict(),
        }

    @staticmethod
    async def test_setup(
        *, model: str, api_base: str, api_key: str | None = None
    ) -> dict:
        """用最小请求验证端点与密钥；不保存任何状态，返回可展示的错误。"""
        api_key = _validate_api_key(api_key)
        candidate = replace(
            MiniConfig(),
            model=(model or "").strip(),
            api_base=(api_base or "").strip(),
        )
        if not api_key:
            api_key, _ = credentials.load_api_key(candidate.api_base)
        if not api_key and candidate.needs_setup():
            return {"ok": False, "error": "请填写 API 密钥"}

        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AsyncOpenAI,
            AuthenticationError,
            NotFoundError,
            RateLimitError,
        )

        client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=candidate.api_base,
            timeout=15.0,
            max_retries=0,
        )
        try:
            response = await client.chat.completions.create(
                model=candidate.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except AuthenticationError:
            return {"ok": False, "error": "鉴权失败（401）：请检查 API 密钥"}
        except NotFoundError:
            return {"ok": False, "error": "模型或端点不存在（404）：请检查模型名与端点地址"}
        except RateLimitError:
            return {"ok": False, "error": "请求被限流（429）：请稍后重试"}
        except (APIConnectionError, APITimeoutError):
            return {"ok": False, "error": "无法连接端点：请检查地址、网络或本地模型服务"}
        except APIStatusError as exc:
            logger.info("设置页连接测试失败: HTTP %s", exc.status_code)
            return {
                "ok": False,
                "error": f"端点返回 HTTP {exc.status_code}，请检查模型与端点配置",
            }
        except Exception as exc:  # noqa: BLE001 - 测试接口只返回安全摘要
            logger.info("设置页连接测试失败: %s", type(exc).__name__)
            return {"ok": False, "error": f"连接失败（{type(exc).__name__}）"}
        finally:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "model": getattr(response, "model", candidate.model)}

    async def _on_setup_request(self, payload: dict) -> dict:
        return await self.apply_setup(
            model=str(payload.get("model") or ""),
            api_base=str(payload.get("api_base") or ""),
            api_key=_as_optional_str(payload.get("api_key")),
        )

    async def _on_setup_test(self, payload: dict) -> dict:
        return await self.test_setup(
            model=str(payload.get("model") or ""),
            api_base=str(payload.get("api_base") or ""),
            api_key=_as_optional_str(payload.get("api_key")),
        )

    @staticmethod
    def _build_subagents(
        config: MiniConfig, provider
    ) -> SubagentManager | None:
        if not config.multi_agent_enabled:
            return None
        return SubagentManager(
            provider,
            max_concurrent=config.max_concurrent_subagents,
            max_iterations=config.max_iterations,
            timeout=config.subagent_timeout,
            max_result_chars=config.subagent_max_result_chars,
        )

    async def start(self) -> None:
        """启动分发器与全部渠道；重复调用幂等。"""
        if self._closed:
            raise RuntimeError("MiniBot 已关闭，不能重新启动")
        if self._started:
            return

        await self.loop.start()
        started: list[tuple[str, Channel]] = []
        try:
            for name, channel in self.channels.items():
                await channel.start()
                started.append((name, channel))
        except Exception:
            logger.exception("渠道启动失败，回滚已启动的渠道")
            for name, channel in reversed(started):
                try:
                    await channel.stop()
                except Exception:
                    logger.exception("回滚渠道 %r 失败", name)
            await self.loop.stop(graceful_timeout=0)
            raise
        self._started = True

    async def stop(self, graceful_timeout: float = 5.0) -> None:
        """优雅停机：先让 loop 处理完排队消息，再关闭渠道与 LLM 客户端。"""
        if self._closed:
            return
        self._closed = True
        self._started = False

        await self.loop.stop(graceful_timeout=graceful_timeout)
        if self.subagents is not None:
            try:
                await self.subagents.close()
            except Exception:
                logger.exception("关闭子代理管理器失败")
        for name, channel in self.channels.items():
            try:
                await channel.stop()
            except Exception:
                logger.exception("停止渠道 %r 失败", name)
        try:
            await self.runner.provider.aclose()
        except Exception:
            logger.exception("关闭 LLM 客户端失败")
        try:
            await self._embedder.aclose()
        except Exception:
            logger.exception("关闭嵌入客户端失败")
        shutdown_tracing()

    async def aclose(self) -> None:
        """``stop()`` 的别名，贴合 ``async with`` / ``aclose`` 习惯。"""
        await self.stop()

    async def __aenter__(self) -> "MiniBot":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
