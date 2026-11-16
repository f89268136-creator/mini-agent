"""部件 5（轻量门面）：Agent —— 单文件即可运行的嵌入式入口。

定位：不想了解总线 / 渠道 / 会话仓库时，用这一个类就够了——
内部只有 list[Message]（记忆）+ AgentRunner（ReAct 循环），
不引入 bus / channels，也不做磁盘持久化；只复用会话层的消息清洗函数，
保证按条数截断后不会把工具调用链切成两半。

    agent = Agent(provider_from_env())
    answer = await agent.chat("现在几点？")

与完整系统的关系：这就是 nanobot 里"一个进程一条会话"的退化形态；
需要多渠道、按会话分片、持久化时，请用 sdk.MiniBot（内部装配
MessageBus + SessionManager + AgentLoop + 渠道，见 sdk.py）。
"""

from __future__ import annotations

from ..config import DEFAULT_SYSTEM_PROMPT
from ..providers import LLMProvider
from ..providers.base import Message
from ..session.manager import sanitize_tool_messages
from .runner import AgentRunner


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 10,
        history_limit: int = 50,
    ):
        # 记忆：整个会话的消息列表，跨多轮保留（多轮对话的关键）
        self.history: list[Message] = [Message(role="system", content=system_prompt)]
        self.runner = AgentRunner(provider, max_iterations=max_iterations)
        # 轻量"记忆上限"：只保留最近 history_limit 条（第一条 system 永远保留）
        self.history_limit = history_limit

    async def chat(self, text: str) -> str:
        # 1) 用户消息进入历史
        self.history.append(Message(role="user", content=text))

        # 2) 超限就截断：system 提示词 + 最近 history_limit 条。
        #    截断可能把 assistant.tool_calls 和对应的 tool 结果分到两边，
        #    必须清洗，否则下一次请求会因孤儿 tool 消息直接 400。
        if len(self.history) > self.history_limit + 1:
            recent = sanitize_tool_messages(self.history[-self.history_limit :])
            self.history = [self.history[0], *recent]

        # 3) runner 在同一份历史上运行（工具调用 / 结果都会追加进来）
        return await self.runner.run(self.history)
