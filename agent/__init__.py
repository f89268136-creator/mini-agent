"""Agent 核心包（对应 nanobot/agent/）。

    runner.py  AgentRunner：一条会话里的 ReAct 循环（调模型 → 工具 → 回灌 → 再调）
    loop.py    AgentLoop：总线 → 按会话分片 → 引擎 → 渠道 的编排
    context.py ContextBuilder：人设 + 长期记忆 + 预算内的短期历史
    request.py RequestContext：把渠道/发送者/会话键安全地传给工具
    subagent.py SubagentManager：delegate / spawn 子代理、限流、超时与结果回收
    agent.py   轻量门面 Agent：单会话 + 记忆，一个类即可嵌入（无总线/渠道）
    tools/     工具功能包：注册表 + 内置工具 + RAG + 长期记忆 + 网络 + 多智能体

导入本包会连带导入 tools 子包，把全部内置工具注册进全局注册表。
"""

from .commands import Command, CommandContext, CommandRegistry
from .request import RequestContext, current_request_context, request_scope
from .runner import AgentRunner
from .loop import AgentLoop
from .context import ContextBuilder
from .skills import Skill, SkillRegistry
from .subagent import (
    SubagentManager,
    SubagentResult,
    SubagentStatus,
    configure_subagent_manager,
    get_subagent_manager,
)
from .agent import Agent
from . import tools  # noqa: F401  # 显式导入即注册全部内置工具

__all__ = [
    "Agent",
    "AgentLoop",
    "AgentRunner",
    "Command",
    "CommandContext",
    "CommandRegistry",
    "ContextBuilder",
    "Skill",
    "SkillRegistry",
    "RequestContext",
    "current_request_context",
    "request_scope",
    "SubagentManager",
    "SubagentResult",
    "SubagentStatus",
    "configure_subagent_manager",
    "get_subagent_manager",
]
