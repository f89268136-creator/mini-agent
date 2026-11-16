"""配置模型与加载器。

优先级（后者覆盖前者）：
    默认值 < 用户配置 < 项目配置 < 环境变量 < 命令行参数

- 用户配置固定在 ``~/.mini_agent/config.json``，一次配置长期生效；
- 项目配置是当前目录的 ``.mini_agent.json``，用于按项目覆盖；
- 显式传入路径（``--config`` / ``MINI_CONFIG``）时只用该文件，不再叠加用户配置；
- 配置文件只保存与默认值不同的键，密钥永远不写入配置文件。

与"能跑就行"的配置不同，这里对每一个字段做类型与取值范围校验：正常启动路径
写错就抛 :class:`ConfigError`；图形界面的修复模式改用
:func:`load_config_report`，把错误收集进报告而不是把用户挡在门外。
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlparse

from .utils import credentials
from .utils.netutil import is_loopback_url

logger = logging.getLogger("mini_agent.config")

DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业可靠、乐于助人的业务 AI 助手。需要实时信息或执行操作时，"
    "必须调用可用工具，并基于真实结果作答，不得虚构数据或执行结果。"
    "默认使用中文，回答时请一步步思考，思考完成后先给结论，"
    "再简要说明依据和可执行建议；信息不足或结果不确定时如实说明。"
)

DEFAULT_CONFIG_FILENAME = ".mini_agent.json"
USER_CONFIG_DIRNAME = ".mini_agent"
USER_CONFIG_FILENAME = "config.json"
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}

# 从网页/聊天窗口复制时常见的零宽字符，肉眼不可见但会破坏 URL 与模型名。
_INVISIBLE_TABLE = {ord(ch): None for ch in "\u200b\u200c\u200d\ufeff"}


def _clean(value: str) -> str:
    return value.translate(_INVISIBLE_TABLE).strip()

_STR_FIELDS = {
    "model",
    "api_base",
    "system_prompt",
    "sessions_dir",
    "memory_dir",
    "memory_backend",
    "mem0_dir",
    "todos_dir",
    "log_level",
    "workspace",
    "knowledge_dir",
    "embedding_backend",
    "tool_approval",
}
_OPTIONAL_STR_FIELDS = {
    "api_key",
    "embedding_model",
    "mem0_embedding_model",
    "langfuse_base_url",
}
_INT_FIELDS = {
    "max_iterations",
    "history_limit",
    "echo_port",
    "web_port",
    "retry_max_attempts",
    "knowledge_chunk_size",
    "knowledge_chunk_overlap",
    "knowledge_top_k",
    "max_input_chars",
    "max_active_sessions",
    "max_concurrent_turns",
    "max_queue_size",
    "memory_max_entries",
    "memory_inject_chars",
    "prompt_max_chars",
    "prompt_max_tokens",
    "context_keep_turns",
    "context_summary_max_chars",
    "max_concurrent_subagents",
    "subagent_max_result_chars",
    "mem0_embedding_dims",
    "todo_max_items",
    "todo_inject_chars",
}
_FLOAT_FIELDS = {
    "retry_base_delay",
    "timeout",
    "turn_timeout",
    "tool_timeout",
    "session_idle_timeout",
    "subagent_timeout",
    "langfuse_sample_rate",
    "knowledge_min_score",
    "knowledge_lexical_weight",
    "knowledge_mmr_lambda",
    "approval_timeout",
    "context_compact_ratio",
}
_BOOL_FIELDS = {
    "allow_shell",
    "allow_private_urls",
    "allow_no_api_key",
    "memory_enabled",
    "todos_enabled",
    "skills_enabled",
    "stream",
    "multi_agent_enabled",
    "langfuse_enabled",
    "langfuse_redact",
}

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """配置不可用。CLI 会把它转换成退出码 2。"""


class MissingCredentialsError(ConfigError):
    """远程端点缺少 API 密钥；CLI 会额外打印首次使用引导。"""


@dataclass
class MiniConfig:
    """全部可配置项。字段名同时充当 JSON 文件里的键名。"""

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str = "https://api.openai.com/v1"
    allow_no_api_key: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_iterations: int = 10
    history_limit: int = 50
    max_input_chars: int = 20_000
    max_active_sessions: int = 200
    # 跨会话真正同时打到 Provider 的回合数上限；0 表示不限。
    max_concurrent_turns: int = 8
    max_queue_size: int = 256
    sessions_dir: str = ".mini_agent_sessions"
    memory_enabled: bool = True
    memory_backend: str = "local"
    memory_dir: str = ".mini_agent_memory"
    mem0_dir: str = ".mini_agent_mem0"
    mem0_embedding_model: str | None = None
    mem0_embedding_dims: int = 1536
    memory_max_entries: int = 200
    memory_inject_chars: int = 4_000
    prompt_max_chars: int = 32_000
    # 估算 token 预算：0 表示只按字符裁剪；超过 ratio 阈值时触发自动压缩。
    prompt_max_tokens: int = 8_000
    context_compact_ratio: float = 0.8
    # 压缩时保留最近多少个用户回合，其余折叠进摘要。
    context_keep_turns: int = 4
    context_summary_max_chars: int = 2_000
    todos_enabled: bool = True
    # 从 ~/.mini_agent/skills 与 <workspace>/.mini_agent_skills 加载提示词技能。
    skills_enabled: bool = True
    todos_dir: str = ".mini_agent_todos"
    todo_max_items: int = 50
    todo_inject_chars: int = 2_000
    stream: bool = True
    multi_agent_enabled: bool = True
    max_concurrent_subagents: int = 4
    subagent_timeout: float = 300.0
    subagent_max_result_chars: int = 8_000
    channels: list[str] = field(default_factory=lambda: ["cli"])
    echo_port: int = 8766
    # Web 前端只监听 127.0.0.1；对外暴露请在前置反向代理上做认证与 TLS。
    web_port: int = 8767
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    timeout: float = 60.0
    # 整轮 ReAct（含工具调用）与单个同步工具的执行上限；0 表示不限。
    turn_timeout: float = 300.0
    tool_timeout: float = 120.0
    log_level: str = "INFO"
    langfuse_enabled: bool = False
    langfuse_base_url: str | None = None
    langfuse_sample_rate: float = 1.0
    # 默认脱敏：不把对话正文、工具参数/结果、检索片段原样上报 Langfuse。
    langfuse_redact: bool = True
    # 有副作用工具（写文件、执行命令、删文档）是否需要用户确认。
    tool_approval: str = "ask"
    approval_timeout: float = 120.0
    workspace: str = "."
    allow_shell: bool = False
    allow_private_urls: bool = False
    session_idle_timeout: float = 900.0
    embedding_model: str | None = None
    embedding_backend: str = "openai"
    knowledge_dir: str = ".mini_agent_knowledge"
    knowledge_chunk_size: int = 500
    knowledge_chunk_overlap: int = 50
    knowledge_top_k: int = 4
    knowledge_min_score: float = 0.0
    knowledge_lexical_weight: float = 0.35
    knowledge_mmr_lambda: float = 0.7
    # 只看不存：记录密钥来自环境变量 / 钥匙串 / 凭据文件，不回显密钥本身。
    credential_source: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """类型检查 -> 规范化 -> 范围检查；失败抛 ConfigError。"""
        self._check_types()
        self._normalize()
        self._check_values()

    def _check_types(self) -> None:
        for name in _STR_FIELDS:
            if not isinstance(getattr(self, name), str):
                raise ConfigError(f"{name} 必须是字符串，当前为 {type(getattr(self, name)).__name__}")
        for name in _OPTIONAL_STR_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"{name} 必须是字符串或 null，当前为 {type(value).__name__}")
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name} 必须是整数，当前为 {value!r}")
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name} 必须是数字，当前为 {value!r}")
        for name in _BOOL_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} 必须是 true/false，当前为 {getattr(self, name)!r}")
        if not isinstance(self.channels, list) or any(
            not isinstance(item, str) for item in self.channels
        ):
            raise ConfigError("channels 必须是字符串数组，例如 [\"cli\", \"echo\"]")

    def _normalize(self) -> None:
        self.model = _clean(self.model)
        self.api_base = _clean(self.api_base).rstrip("/")
        self.sessions_dir = self.sessions_dir.strip()
        self.memory_dir = self.memory_dir.strip()
        self.memory_backend = self.memory_backend.strip().lower()
        self.mem0_dir = self.mem0_dir.strip()
        self.todos_dir = self.todos_dir.strip()
        self.knowledge_dir = self.knowledge_dir.strip()
        self.embedding_backend = self.embedding_backend.strip().lower()
        self.tool_approval = self.tool_approval.strip().lower()
        self.workspace = self.workspace.strip()
        self.log_level = self.log_level.strip().upper()
        if self.api_key is not None:
            self.api_key = self.api_key.strip() or None
        if self.embedding_model is not None:
            self.embedding_model = self.embedding_model.strip() or None
        if self.mem0_embedding_model is not None:
            self.mem0_embedding_model = self.mem0_embedding_model.strip() or None
        if self.langfuse_base_url is not None:
            self.langfuse_base_url = self.langfuse_base_url.strip().rstrip("/") or None
        self.channels = list(dict.fromkeys(item.strip() for item in self.channels if item.strip()))
        self.retry_base_delay = float(self.retry_base_delay)
        self.timeout = float(self.timeout)
        self.turn_timeout = float(self.turn_timeout)
        self.tool_timeout = float(self.tool_timeout)
        self.session_idle_timeout = float(self.session_idle_timeout)
        self.subagent_timeout = float(self.subagent_timeout)
        self.langfuse_sample_rate = float(self.langfuse_sample_rate)
        self.approval_timeout = float(self.approval_timeout)
        self.context_compact_ratio = float(self.context_compact_ratio)

    def _check_values(self) -> None:
        if not self.model:
            raise ConfigError("model 不能为空")
        parsed_base = urlparse(self.api_base)
        if parsed_base.scheme not in ("http", "https") or not parsed_base.netloc:
            raise ConfigError(
                f"api_base 必须是包含主机名的 http:// 或 https:// 地址，"
                f"当前为 {self.api_base!r}"
            )
        if parsed_base.username or parsed_base.password:
            raise ConfigError(
                "api_base 不能内嵌用户名/密码，请把密钥放到 api_key 或 OPENAI_API_KEY"
            )
        if not self.sessions_dir:
            raise ConfigError("sessions_dir 不能为空")
        if not self.memory_dir:
            raise ConfigError("memory_dir 不能为空")
        if self.memory_backend not in ("local", "mem0"):
            raise ConfigError(
                f"memory_backend 必须是 local 或 mem0，当前为 {self.memory_backend!r}"
            )
        if not self.mem0_dir:
            raise ConfigError("mem0_dir 不能为空")
        if not self.todos_dir:
            raise ConfigError("todos_dir 不能为空")
        if not self.knowledge_dir:
            raise ConfigError("knowledge_dir 不能为空")
        if self.embedding_backend not in ("openai", "hashing"):
            raise ConfigError(
                f"embedding_backend 必须是 openai 或 hashing，当前为 {self.embedding_backend!r}"
            )
        if self.tool_approval not in ("ask", "off"):
            raise ConfigError(
                f"tool_approval 必须是 ask 或 off，当前为 {self.tool_approval!r}"
            )
        if not math.isfinite(self.approval_timeout) or not 1 <= self.approval_timeout <= 600:
            raise ConfigError("approval_timeout 必须在 1..600 秒之间")
        if not 1 <= self.mem0_embedding_dims <= 65536:
            raise ConfigError("mem0_embedding_dims 必须在 1..65536 之间")
        if not self.system_prompt:
            raise ConfigError("system_prompt 不能为空")
        if not self.workspace:
            raise ConfigError("workspace 不能为空")
        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"log_level 必须是 {', '.join(sorted(VALID_LOG_LEVELS))} 之一，"
                f"当前为 {self.log_level!r}"
            )
        if not 1 <= self.max_iterations <= 100:
            raise ConfigError("max_iterations 必须在 1..100 之间")
        if not 1 <= self.history_limit <= 10000:
            raise ConfigError("history_limit 必须在 1..10000 之间")
        if not 1 <= self.max_input_chars <= 1_000_000:
            raise ConfigError("max_input_chars 必须在 1..1000000 之间")
        if not 1 <= self.max_active_sessions <= 100_000:
            raise ConfigError("max_active_sessions 必须在 1..100000 之间")
        if not 0 <= self.max_concurrent_turns <= 100_000:
            raise ConfigError("max_concurrent_turns 必须在 0..100000 之间（0 表示不限）")
        if not 1 <= self.max_queue_size <= 100_000:
            raise ConfigError("max_queue_size 必须在 1..100000 之间")
        if not 1 <= self.memory_max_entries <= 100_000:
            raise ConfigError("memory_max_entries 必须在 1..100000 之间")
        if not 0 <= self.memory_inject_chars <= 1_000_000:
            raise ConfigError("memory_inject_chars 必须在 0..1000000 之间（0 表示不注入）")
        if not 1 <= self.todo_max_items <= 500:
            raise ConfigError("todo_max_items 必须在 1..500 之间")
        if not 0 <= self.todo_inject_chars <= 1_000_000:
            raise ConfigError("todo_inject_chars 必须在 0..1000000 之间（0 表示不注入）")
        if not 100 <= self.prompt_max_chars <= 2_000_000:
            raise ConfigError("prompt_max_chars 必须在 100..2000000 之间")
        if not 0 <= self.prompt_max_tokens <= 1_000_000:
            raise ConfigError("prompt_max_tokens 必须在 0..1000000 之间（0 表示关闭 token 预算）")
        if (
            not math.isfinite(self.context_compact_ratio)
            or not 0.1 <= self.context_compact_ratio <= 1
        ):
            raise ConfigError("context_compact_ratio 必须在 0.1..1 之间")
        if not 1 <= self.context_keep_turns <= 50:
            raise ConfigError("context_keep_turns 必须在 1..50 之间")
        if not 200 <= self.context_summary_max_chars <= 20_000:
            raise ConfigError("context_summary_max_chars 必须在 200..20000 之间")
        if not 1 <= self.max_concurrent_subagents <= 64:
            raise ConfigError("max_concurrent_subagents 必须在 1..64 之间")
        if not 1 <= self.subagent_timeout <= 3600:
            raise ConfigError("subagent_timeout 必须在 1..3600 秒之间")
        if not 500 <= self.subagent_max_result_chars <= 100_000:
            raise ConfigError("subagent_max_result_chars 必须在 500..100000 之间")
        if not 1 <= self.echo_port <= 65535:
            raise ConfigError("echo_port 必须在 1..65535 之间")
        if not 1 <= self.web_port <= 65535:
            raise ConfigError("web_port 必须在 1..65535 之间")
        if (
            "echo" in self.channels
            and "web" in self.channels
            and self.echo_port == self.web_port
        ):
            raise ConfigError("echo_port 与 web_port 不能相同")
        if not 1 <= self.retry_max_attempts <= 10:
            raise ConfigError("retry_max_attempts 必须在 1..10 之间")
        if not 0 <= self.retry_base_delay <= 60:
            raise ConfigError("retry_base_delay 必须在 0..60 秒之间")
        if not math.isfinite(self.timeout) or not 0.1 <= self.timeout <= 600:
            raise ConfigError("timeout 必须在 0.1..600 秒之间")
        if not math.isfinite(self.turn_timeout) or not 0 <= self.turn_timeout <= 86_400:
            raise ConfigError("turn_timeout 必须在 0..86400 秒之间（0 表示不限）")
        if not math.isfinite(self.tool_timeout) or not 0 <= self.tool_timeout <= 3_600:
            raise ConfigError("tool_timeout 必须在 0..3600 秒之间（0 表示不限）")
        if not math.isfinite(self.session_idle_timeout) or not 0 <= self.session_idle_timeout <= 86400:
            raise ConfigError("session_idle_timeout 必须在 0..86400 秒之间（0 表示不回收）")
        if not 51 <= self.knowledge_chunk_size <= 10000:
            raise ConfigError("knowledge_chunk_size 必须在 51..10000 之间")
        if not 0 <= self.knowledge_chunk_overlap < self.knowledge_chunk_size:
            raise ConfigError("knowledge_chunk_overlap 必须在 0..knowledge_chunk_size-1 之间")
        if not 1 <= self.knowledge_top_k <= 20:
            raise ConfigError("knowledge_top_k 必须在 1..20 之间")
        if not math.isfinite(self.knowledge_min_score) or not 0 <= self.knowledge_min_score <= 1:
            raise ConfigError("knowledge_min_score 必须在 0..1 之间")
        if (
            not math.isfinite(self.knowledge_lexical_weight)
            or not 0 <= self.knowledge_lexical_weight <= 1
        ):
            raise ConfigError("knowledge_lexical_weight 必须在 0..1 之间")
        if not math.isfinite(self.knowledge_mmr_lambda) or not 0 <= self.knowledge_mmr_lambda <= 1:
            raise ConfigError("knowledge_mmr_lambda 必须在 0..1 之间")
        if not math.isfinite(self.langfuse_sample_rate) or not 0 <= self.langfuse_sample_rate <= 1:
            raise ConfigError("langfuse_sample_rate 必须在 0..1 之间")
        if self.langfuse_base_url is not None:
            parsed_trace = urlparse(self.langfuse_base_url)
            if (
                parsed_trace.scheme not in ("http", "https")
                or not parsed_trace.netloc
                or parsed_trace.username
                or parsed_trace.password
            ):
                raise ConfigError(
                    "langfuse_base_url 必须是包含主机名的 http:// 或 https:// 地址，且不能内嵌凭据"
                )

    def validate_credentials(self) -> None:
        """启动期凭据检查。

        远程端点缺少密钥时，真实网络请求通常要等到超时或收到 401 才会暴露问题。
        与其让用户面对"卡住不动"，不如在构造阶段直接失败并给出可操作的提示。
        本地端点（127.0.0.1 / localhost）默认允许无密钥，适配 Ollama 等本地模型；
        其他场景可用 ``allow_no_api_key=true`` 显式豁免。
        """
        _warn_insecure_http(self.api_base)
        if self.langfuse_enabled:
            missing = [
                name
                for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
                if not os.environ.get(name)
            ]
            if missing:
                raise MissingCredentialsError(
                    "已启用 Langfuse，但缺少环境变量："
                    + "、".join(missing)
                    + "。密钥只能通过环境变量提供，不能写入配置文件。"
                )
        if self.api_key or self.allow_no_api_key:
            return
        if is_loopback_url(self.api_base):
            return
        raise MissingCredentialsError(
            "未配置 API 密钥：钥匙串、OPENAI_API_KEY 与凭据文件均为空，"
            f"而端点 {self.api_base} 需要鉴权。运行 mini-agent login 完成一次配置。"
        )

    def needs_setup(self) -> bool:
        """还缺端点或凭据时，界面应引导用户进入设置页而不是直接报错退出。"""
        if self.allow_no_api_key or is_loopback_url(self.api_base):
            return False
        return not self.api_key

    def describe(self) -> dict[str, Any]:
        """返回可安全打印/记录的配置快照（密钥脱敏）。"""
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        if data.get("api_key"):
            data["api_key"] = "***"
        return data


_warned_insecure_bases: set[str] = set()


def _warn_insecure_http(api_base: str) -> None:
    """远程明文 HTTP 会泄露密钥与对话内容；只告警一次，避免日志刷屏。"""
    if not api_base.startswith("http://") or is_loopback_url(api_base):
        return
    if api_base in _warned_insecure_bases:
        return
    _warned_insecure_bases.add(api_base)
    logger.warning(
        "api_base 使用明文 HTTP，API 密钥与对话内容可能被同网段窃听: %s", api_base
    )


def _warn_secret_hygiene(path: str, data: dict[str, Any]) -> None:
    """配置文件里出现密钥或权限过宽时给出告警（不打印密钥本身）。"""
    if data.get("api_key"):
        logger.warning(
            "配置文件 %s 中包含 api_key，建议改用系统钥匙串（mini-agent login）"
            "或环境变量 OPENAI_API_KEY",
            path,
        )
    if os.name == "nt":
        return
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return
    if mode & 0o077:
        logger.warning(
            "配置文件 %s 权限过宽（%o），同机其他用户可能读到其中的密钥；建议 chmod 600",
            path,
            mode & 0o777,
        )


@dataclass(frozen=True)
class ConfigIssue:
    """一条配置问题；error 会阻止正常启动，warning 只提示。"""

    source: str
    message: str
    severity: str = "error"


@dataclass
class ConfigReport:
    """加载结果：配置本身 + 实际读取的文件 + 问题清单 + 密钥来源。

    修复模式（Web 设置页）用它判断该显示"可直接使用"还是"需要修复"，
    并展示配置来自哪些文件；``issues`` 中绝不包含密钥明文。
    """

    config: MiniConfig
    paths: list[str]
    issues: list[ConfigIssue]

    @property
    def needs_repair(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.severity == "error"]


def user_config_path() -> str:
    """用户级配置：一次配置，所有项目共用。"""
    return os.path.join(
        os.path.expanduser("~"), USER_CONFIG_DIRNAME, USER_CONFIG_FILENAME
    )


def project_config_path() -> str:
    """项目级配置：当前目录，用于按项目覆盖用户配置。"""
    return DEFAULT_CONFIG_FILENAME


def resolve_config_paths(path: str | None = None) -> list[str]:
    """按优先级从低到高返回实际要读取的文件。

    显式路径（``--config`` / ``MINI_CONFIG``）表示"只用这个文件"，
    避免部署脚本以为自己控制了配置、结果又被用户目录里的旧值覆盖。
    """
    explicit = path or os.environ.get("MINI_CONFIG")
    if explicit:
        return [explicit]
    candidates = [user_config_path(), project_config_path()]
    return [item for item in candidates if os.path.exists(item)]


def load_config(
    path: str | None = None, *, require_credentials: bool = True
) -> MiniConfig:
    """正常启动路径：任何配置错误都抛 ConfigError，避免带病运行。

    读取顺序为 默认值 -> 用户配置 -> 项目配置 -> 环境变量。
    显式传入 ``path`` 时只看该文件；文件必须存在。
    """
    report = load_config_report(path)
    if report.needs_repair:
        raise ConfigError("；".join(report.errors))
    cfg = report.config
    if require_credentials:
        cfg.validate_credentials()
    return cfg


def load_config_report(path: str | None = None) -> ConfigReport:
    """宽松加载：坏文件只记录问题，仍返回一份可用配置供修复界面启动。

    - JSON 损坏、类型/范围错误：记 error，并用默认值 + 环境变量重建配置；
    - 未知键：记 warning 后忽略（向前兼容）；
    - 凭据按 环境变量 -> 钥匙串 -> 凭据文件 的顺序注入，只记录来源。
    """
    cfg = MiniConfig()
    paths: list[str] = []
    issues: list[ConfigIssue] = []
    merged: dict[str, Any] = {}
    valid_fields = {item.name for item in fields(MiniConfig)} - {"credential_source"}

    for candidate in resolve_config_paths(path):
        if not os.path.exists(candidate):
            issues.append(ConfigIssue(candidate, f"配置文件不存在: {candidate}"))
            continue
        paths.append(candidate)
        try:
            data = _load_json_file(candidate)
        except ConfigError as exc:
            issues.append(ConfigIssue(candidate, str(exc)))
            continue
        _warn_secret_hygiene(candidate, data)
        for key, value in data.items():
            if key in valid_fields:
                merged[key] = value
            else:
                logger.warning("配置文件中的未知键已忽略: %s", key)
                issues.append(
                    ConfigIssue(candidate, f"未知配置键已忽略: {key}", "warning")
                )

    for key, value in merged.items():
        setattr(cfg, key, value)

    try:
        _apply_env(cfg)
        cfg.validate()
    except ConfigError as exc:
        source = "、".join(paths) or "<默认值>"
        issues.append(ConfigIssue(source, str(exc)))
        cfg = _fallback_config(issues)

    key, source = credentials.load_api_key(cfg.api_base)
    if key:
        cfg.api_key = key
        cfg.credential_source = source
    elif cfg.api_key:
        cfg.credential_source = "config"
    return ConfigReport(config=cfg, paths=paths, issues=issues)


def load_file_config(path: str | None = None) -> MiniConfig:
    """只装配 JSON 文件层（不含环境变量与凭据），作为写回磁盘的基准。

    不能拿运行中的配置做写入基准：CLI 的 ``--port``、强制 ``web`` 渠道等
    运行时覆盖一旦落盘，下次启动就会意外生效。无法解析的文件按空配置处理，
    因为它随后会被本次写入整体替换（修复模式依赖这一点）。
    """
    cfg = MiniConfig()
    valid_fields = {item.name for item in fields(MiniConfig)} - {"credential_source"}
    for candidate in resolve_config_paths(path):
        if not os.path.exists(candidate):
            continue
        try:
            data = _load_json_file(candidate)
        except ConfigError:
            logger.warning("写回配置时忽略无法解析的文件: %s", candidate)
            continue
        for key, value in data.items():
            if key in valid_fields:
                setattr(cfg, key, value)
    cfg.validate()
    return cfg


def _fallback_config(issues: list[ConfigIssue]) -> MiniConfig:
    """坏配置不能把用户挡在修复页面外：退回默认值，再尽量保留合法环境变量。"""
    cfg = MiniConfig()
    try:
        _apply_env(cfg)
        cfg.validate()
    except ConfigError as exc:
        issues.append(ConfigIssue("<环境变量>", str(exc)))
        cfg = MiniConfig()
    return cfg


def config_diff(config: MiniConfig) -> dict[str, Any]:
    """只保留与默认值不同的键；``api_key`` 永不写入配置文件。"""
    defaults = MiniConfig()
    data: dict[str, Any] = {}
    for item in fields(MiniConfig):
        if item.name in ("api_key", "credential_source"):
            continue
        value = getattr(config, item.name)
        if value != getattr(defaults, item.name):
            data[item.name] = value
    return data


def write_config(config: MiniConfig, path: str | None = None) -> str:
    """原子写入与默认值的差异；返回实际路径。

    默认目标是：已有项目配置则续写项目配置，否则写用户配置。
    权限收紧到 0600（Windows 上尽力收紧 ACL），因为即便没有密钥，
    配置里也可能包含工作区、端点等敏感信息。
    """
    target = path or os.environ.get("MINI_CONFIG") or _default_write_path()
    data = config_diff(config)
    parent = os.path.dirname(os.path.abspath(target)) or "."
    os.makedirs(parent, exist_ok=True, mode=0o700)
    fd, temp_path = tempfile.mkstemp(prefix=".config-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        if os.name == "nt":
            credentials._restrict_windows_acl(temp_path)
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return target


def _default_write_path() -> str:
    project = project_config_path()
    if os.path.exists(project):
        return project
    return user_config_path()


def _load_json_file(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"配置文件 {path} 不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件 {path} 的顶层必须是 JSON 对象")
    return data


def _apply_env(cfg: MiniConfig) -> None:
    str_vars = {
        "MODEL": "model",
        "OPENAI_API_KEY": "api_key",
        "OPENAI_BASE_URL": "api_base",
        "MINI_SESSIONS_DIR": "sessions_dir",
        "MINI_MEMORY_DIR": "memory_dir",
        "MINI_MEMORY_BACKEND": "memory_backend",
        "MINI_MEM0_DIR": "mem0_dir",
        "MINI_TODOS_DIR": "todos_dir",
        "MINI_KNOWLEDGE_DIR": "knowledge_dir",
        "MINI_LOG_LEVEL": "log_level",
        "MINI_WORKSPACE": "workspace",
        "MINI_TOOL_APPROVAL": "tool_approval",
    }
    for env_name, field_name in str_vars.items():
        value = os.environ.get(env_name)
        if value:
            setattr(cfg, field_name, value)

    embedding_model = os.environ.get("MINI_EMBEDDING_MODEL") or os.environ.get("EMBEDDING_MODEL")
    if embedding_model:
        cfg.embedding_model = embedding_model

    embedding_backend = (
        os.environ.get("MINI_EMBEDDING_BACKEND")
        or os.environ.get("EMBEDDING_BACKEND")
    )
    if embedding_backend:
        cfg.embedding_backend = embedding_backend

    optional_str_vars = {
        "MINI_MEM0_EMBEDDING_MODEL": "mem0_embedding_model",
        "MINI_LANGFUSE_BASE_URL": "langfuse_base_url",
    }
    for env_name, field_name in optional_str_vars.items():
        value = os.environ.get(env_name)
        if value:
            setattr(cfg, field_name, value)

    raw_channels = os.environ.get("MINI_CHANNELS")
    if raw_channels is not None:
        cfg.channels = [item.strip() for item in raw_channels.split(",") if item.strip()]

    bool_vars = {
        "MINI_ALLOW_SHELL": "allow_shell",
        "MINI_ALLOW_PRIVATE_URLS": "allow_private_urls",
        "MINI_ALLOW_NO_API_KEY": "allow_no_api_key",
        "MINI_MEMORY_ENABLED": "memory_enabled",
        "MINI_TODOS_ENABLED": "todos_enabled",
        "MINI_SKILLS_ENABLED": "skills_enabled",
        "MINI_STREAM": "stream",
        "MINI_MULTI_AGENT_ENABLED": "multi_agent_enabled",
        "MINI_LANGFUSE_ENABLED": "langfuse_enabled",
        "MINI_LANGFUSE_REDACT": "langfuse_redact",
    }
    for env_name, field_name in bool_vars.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            setattr(cfg, field_name, _parse_bool(env_name, raw))

    int_vars = {
        "MAX_ITERATIONS": "max_iterations",
        "HISTORY_LIMIT": "history_limit",
        "ECHO_PORT": "echo_port",
        "MINI_WEB_PORT": "web_port",
        "MINI_RETRY_MAX_ATTEMPTS": "retry_max_attempts",
        "MINI_KNOWLEDGE_CHUNK_SIZE": "knowledge_chunk_size",
        "MINI_KNOWLEDGE_CHUNK_OVERLAP": "knowledge_chunk_overlap",
        "MINI_KNOWLEDGE_TOP_K": "knowledge_top_k",
        "MINI_MAX_INPUT_CHARS": "max_input_chars",
        "MINI_MAX_ACTIVE_SESSIONS": "max_active_sessions",
        "MINI_MAX_CONCURRENT_TURNS": "max_concurrent_turns",
        "MINI_MAX_QUEUE_SIZE": "max_queue_size",
        "MINI_MEMORY_MAX_ENTRIES": "memory_max_entries",
        "MINI_MEM0_EMBEDDING_DIMS": "mem0_embedding_dims",
        "MINI_MEMORY_INJECT_CHARS": "memory_inject_chars",
        "MINI_PROMPT_MAX_CHARS": "prompt_max_chars",
        "MINI_PROMPT_MAX_TOKENS": "prompt_max_tokens",
        "MINI_CONTEXT_KEEP_TURNS": "context_keep_turns",
        "MINI_CONTEXT_SUMMARY_MAX_CHARS": "context_summary_max_chars",
        "MINI_TODO_MAX_ITEMS": "todo_max_items",
        "MINI_TODO_INJECT_CHARS": "todo_inject_chars",
        "MINI_MAX_CONCURRENT_SUBAGENTS": "max_concurrent_subagents",
        "MINI_SUBAGENT_MAX_RESULT_CHARS": "subagent_max_result_chars",
    }
    for env_name, field_name in int_vars.items():
        _set_number(cfg, env_name, field_name, os.environ.get(env_name), int)

    float_vars = {
        "MINI_RETRY_BASE_DELAY": "retry_base_delay",
        "MINI_TIMEOUT": "timeout",
        "MINI_TURN_TIMEOUT": "turn_timeout",
        "MINI_TOOL_TIMEOUT": "tool_timeout",
        "MINI_SESSION_IDLE_TIMEOUT": "session_idle_timeout",
        "MINI_SUBAGENT_TIMEOUT": "subagent_timeout",
        "MINI_LANGFUSE_SAMPLE_RATE": "langfuse_sample_rate",
        "MINI_KNOWLEDGE_MIN_SCORE": "knowledge_min_score",
        "MINI_KNOWLEDGE_LEXICAL_WEIGHT": "knowledge_lexical_weight",
        "MINI_KNOWLEDGE_MMR_LAMBDA": "knowledge_mmr_lambda",
        "MINI_APPROVAL_TIMEOUT": "approval_timeout",
        "MINI_CONTEXT_COMPACT_RATIO": "context_compact_ratio",
    }
    for env_name, field_name in float_vars.items():
        _set_number(cfg, env_name, field_name, os.environ.get(env_name), float)


def _parse_bool(env_name: str, raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigError(f"环境变量 {env_name} 必须是 true/false，当前为 {raw!r}")


def _set_number(
    cfg: MiniConfig, env_name: str, field_name: str, raw: str | None, cast: type
) -> None:
    if raw is None or raw == "":
        return
    try:
        setattr(cfg, field_name, cast(raw))
    except ValueError as exc:
        kind = "整数" if cast is int else "数字"
        raise ConfigError(f"环境变量 {env_name} 必须是合法{kind}，当前为 {raw!r}") from exc
