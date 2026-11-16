"""mini_agent：面向生产的轻量异步 AI Agent 运行时。

设计参考 nanobot 的核心架构（MessageBus、按会话分片、ReAct、JSONL 会话），
但以"可直接部署的小型运行时"为目标：配置强校验、日志、工具安全边界、
原子持久化、优雅停机与并发串行化都是默认行为，而不是可选项。

目录结构::

    providers/      模型层：数据结构 / OpenAI 兼容 provider / 嵌入模型
    agent/          Agent 核心：ReAct runner、消息编排 loop、提示词拼装 context
    agent/tools/    工具：注册表、内置工具、RAG、长期记忆、Web、多智能体、安全策略
    bus/            消息总线：渠道与内核解耦
    session/        会话仓库：JSONL 增量持久化 + 会话级任务清单（todo）
    memory/         长期记忆：按会话隔离的 JSONL 记忆
    channels/       渠道：CLI / Echo / Web（内置单页前端）
    runtime/        进程运行时：启动预检、单实例复用
    utils/          通用工具：错误翻译、命名、URL 判定、日志、追踪、凭据
    config.py       配置模型与加载器（JSON + 环境变量 + 严格校验）
    sdk.py          MiniBot 门面：装配以上全部
    __main__.py     命令行入口

两种主要用法::

    # 完整系统（渠道 + 会话分片 + 持久化 + 多智能体）
    from mini_agent import MiniBot
    bot = MiniBot.from_config()
    print(await bot.chat("你好"))

    # 最简模式（单会话，无总线/渠道）
    from mini_agent import Agent
    from mini_agent.providers import provider_from_env
    agent = Agent(provider_from_env())
    print(await agent.chat("你好"))
"""

from __future__ import annotations

import logging

__version__ = "1.0.0"

# 库代码默认不向终端输出日志；应用层可通过 setup_logging() 启用。
logging.getLogger("mini_agent").addHandler(logging.NullHandler())

# 导入顺序：providers（无依赖）→ agent（触发工具注册）→ 基础设施 → 门面。
from . import providers
from . import agent

from .bus import InboundMessage, MessageBus, OutboundMessage
from .channels import Channel, CliChannel, EchoChannel, WebChannel, build_channels
from .config import ConfigError, MiniConfig, load_config
from .memory import MemoryEntry, MemoryStore, configure_memory_store
from .session import Session, SessionManager
from .utils.observability import setup_logging

from .agent import (
    Agent,
    AgentLoop,
    AgentRunner,
    ContextBuilder,
    RequestContext,
    SubagentManager,
    SubagentResult,
    SubagentStatus,
)
from .sdk import MiniBot

from .providers import (
    LLMProvider,
    OpenAICompatProvider,
    RetryingProvider,
    build_embedder,
    build_provider,
)

__all__ = [
    "__version__",
    "ConfigError",
    "setup_logging",
    "LLMProvider",
    "OpenAICompatProvider",
    "RetryingProvider",
    "build_provider",
    "build_embedder",
    "MessageBus",
    "InboundMessage",
    "OutboundMessage",
    "Session",
    "SessionManager",
    "MemoryEntry",
    "MemoryStore",
    "configure_memory_store",
    "MiniConfig",
    "load_config",
    "Channel",
    "CliChannel",
    "EchoChannel",
    "WebChannel",
    "build_channels",
    "AgentRunner",
    "AgentLoop",
    "ContextBuilder",
    "RequestContext",
    "SubagentManager",
    "SubagentResult",
    "SubagentStatus",
    "Agent",
    "MiniBot",
]
