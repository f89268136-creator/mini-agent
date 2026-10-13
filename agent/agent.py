"""部件 5（最简门面）：Agent —— "1 个文件就能跑"的教学入口。

定位：不想了解总线 / 渠道 / 会话仓库时，用这一个类就够了——
内部只有 list[Message]（记忆）+ AgentRunner（ReAct 循环），
不引入 bus / sessions / channels 任何文件。

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
        # 教学版最简"记忆上限"：只保留最近 history_limit 条（第一条 system 永远保留）
        self.history_limit = history_limit

    async def chat(self, text: str) -> str:
        # 1) 用户消息进入历史
        self.history.append(Message(role="user", content=text))

        # 2) 超限就截断：system 提示词 + 最近 history_limit 条。
        #    （完整系统里这一步是 SessionManager 的职责，见 session/manager.py 与 loop.py。）
        if len(self.history) > self.history_limit + 1:
            self.history = [self.history[0], *self.history[-self.history_limit:]]

        # 3) runner 在同一份历史上运行（工具调用 / 结果都会追加进来）
        return await self.runner.run(self.history)
