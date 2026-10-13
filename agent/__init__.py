"""Agent 核心包（对应 nanobot/agent/）。

    runner.py  AgentRunner：一条会话里的 ReAct 循环（调模型 → 工具 → 回灌 → 再调）
    loop.py    AgentLoop：总线 → 按会话分片 → 引擎 → 渠道 的编排
    agent.py   最简门面 Agent：单会话 + 记忆，一个类就能跑（逃课版）
    tools/     工具功能包：注册表 + 内置工具 + RAG + 网络

导入本包会连带导入 tools 子包，把全部内置工具注册进全局注册表。
"""

from .runner import AgentRunner
from .loop import AgentLoop
from .agent import Agent
from . import tools  # noqa: F401  # 显式导入即注册全部内置工具

__all__ = ["Agent", "AgentLoop", "AgentRunner"]
