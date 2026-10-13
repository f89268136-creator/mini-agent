"""mini_agent：参考 nanobot 精简而成的最小可用 Agent（asyncio 版）。

目录结构（每个子包对应 nanobot 的同名目录，括号内是 nanobot 中的位置）：

    providers/      模型功能包（providers/）
        base.py           数据结构：消息 / 工具调用 / 模型响应 + LLMProvider 抽象
        openai_compat.py  OpenAI 兼容提供者（AsyncOpenAI）+ RetryingProvider 重试包装
        embeddings.py     嵌入模型抽象 + OpenAI 兼容实现（providers/ 的 embeddings）
    agent/          Agent 核心包（agent/）
        runner.py       ReAct 引擎循环（agent/runner.py）
        loop.py         编排：总线 → 按会话分片 → 引擎 → 渠道（agent/loop.py）
        agent.py        最简门面：单会话 + 记忆（agent/loop.py 的退化形态）
        tools/          工具功能包（agent/tools/）
            registry.py    注册表 + @register_tool + schema 自动生成
            builtin.py     7 个基础工具：时间 / 计算器 / 文件读写 / 目录 / 命令
            knowledge.py   RAG 知识库：切块 / 向量检索 + 知识库工具
            web.py         网络搜索 + 网页抓取
    bus/            消息总线：渠道与内核解耦（bus/queue.py）
    session/        会话仓库：按会话键存历史、截断、落盘（session/）
    channels/       渠道：CLI / Echo（channels/）
    config.py       配置：默认值 + 环境变量 / 配置文件加载（config/）
    sdk.py          完整门面 MiniBot：装配以上全部（nanobot.py）
    __main__.py     命令行入口（cli/）

导入本包即会注册全部工具（基础工具 + RAG + 网络工具）。

两种用法：
    # 1) 最简：一个类跑起来（无总线 / 无渠道 / 单会话）
    from mini_agent import Agent
    answer = await Agent(provider_from_env()).chat("你好")

    # 2) 完整系统：总线 + 会话分片 + 多渠道（推荐）
    from mini_agent import MiniBot
    answer = await MiniBot.from_config().chat("你好")
"""

# 导入顺序有依赖，不要随意调整：
#   1) providers：无依赖的模型层（数据结构 / 对话模型 / 嵌入模型）；
#   2) agent：核心包——导入 agent.runner 时会连带导入 agent.tools，
#      把内置工具全部注册进全局注册表（tools 各子模块在导入时自我注册）；
#   3) 基础设施：总线 / 配置 / 会话 / 渠道；
#   4) 门面：MiniBot 装配以上全部。
# 这里全是纯模块导入：不会创建 LLM 客户端、不会起后台任务、不会建立连接；
# 真正建客户端的时刻是 MiniBot(...) / build_provider(config) / provider_from_env()。
from . import providers
from . import agent

from .bus import InboundMessage, MessageBus, OutboundMessage
from .channels import Channel, CliChannel, EchoChannel, build_channels
from .config import MiniConfig, load_config
from .session import Session, SessionManager

from .agent import Agent, AgentLoop, AgentRunner
from .sdk import MiniBot

from .providers import LLMProvider, OpenAICompatProvider, RetryingProvider
from .providers.base import LLMResponse, Message, ToolCall

__all__ = [
    # 数据结构 / provider
    "LLMResponse",
    "Message",
    "ToolCall",
    "LLMProvider",
    "OpenAICompatProvider",
    "RetryingProvider",
    # 总线
    "MessageBus",
    "InboundMessage",
    "OutboundMessage",
    # 会话 / 配置
    "Session",
    "SessionManager",
    "MiniConfig",
    "load_config",
    # 渠道
    "Channel",
    "CliChannel",
    "EchoChannel",
    "build_channels",
    # 编排层
    "AgentRunner",
    "AgentLoop",
    "Agent",
    "MiniBot",
]
