"""部件 6（编排层）：AgentRunner —— 引擎的 ReAct 循环（对应 nanobot/agent/runner.py）。

每一轮做的事：
    1) 把全部消息发给模型（async 调用，等它返回）；
    2) 把模型回复（可能含工具调用）追加进消息列表；
    3) 如果模型没有请求工具 → 这就是最终答案，结束；
    4) 如果请求了工具 → 逐个执行，把结果作为 role="tool" 的消息回灌，回到第 1 步。

与旧版（同步）相比的两处升级：
    - provider.chat 是 async 的，这里用 await 调用，不阻塞事件循环；
    - 工具本身仍是同步函数，用 asyncio.to_thread 扔进线程池执行（见 _execute）。

分工说明（对应 nanobot 的架构）：
    runner 只管"这一条会话的 ReAct 循环"——它不知道有总线、有渠道、有多个会话；
    那些"谁把消息送进来、回复发去哪"的编排工作在 loop.py 里完成。
"""

from __future__ import annotations

import asyncio
import json

from ..providers import LLMProvider
from ..providers.base import Message, ToolCall
from .tools import all_tool_schemas, get_tool


class AgentRunner:
    def __init__(self, provider: LLMProvider, max_iterations: int = 10):
        self.provider = provider
        self.max_iterations = max_iterations

    async def run(self, messages: list[Message]) -> str:
        """在 messages 上跑 ReAct 循环（会原地往 messages 里追加内容），返回最终文本。

        注意：入参是"某一条会话的历史快照"，run 结束后里面会多出
        assistant / tool 消息——调用方（AgentLoop）只挑选最终答案写回会话存储。
        """
        for _ in range(self.max_iterations):
            # 1) 调模型（await：请求期间事件循环可以去处理别的会话的消息）
            response = await self.provider.chat(messages, tools=all_tool_schemas())

            # 2) 追加 assistant 消息（即使含工具调用也要原样存，这是接口要求）
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls or None,
                )
            )

            # 3) 判定一：没有工具调用 → 这一轮就是最终答案
            if not response.has_tool_calls:
                return response.content or ""

            # 3b) 判定二（对应 nanobot 的 LLMResponse.should_execute_tools）：
            #     finish_reason 属于 refusal / content_filter / error 时，即使响应里
            #     带着 tool_calls，也不能执行——那往往是网关/安全过滤"借模型之口"
            #     注入的调用（nanobot 源码注释指向 issue #3220 的网关注入 bug）。
            #     不看 finish_reason 就信工具调用字段，会被这类注入骗执行工具。
            if not response.should_execute_tools:
                return response.content or ""

            # 4) 有工具调用 → 逐个执行，并把结果回灌
            for call in response.tool_calls:
                result = await self._execute(call)
                messages.append(
                    Message(role="tool", content=result, tool_call_id=call.id)
                )
            # 循环回到顶部：模型看到工具结果后继续推理

        return "（已达到最大工具轮次，仍未得到最终答案）"

    async def _execute(self, call: ToolCall) -> str:
        """执行单次工具调用：找工具 → 解析参数 → 运行。"""
        tool = get_tool(call.name)
        if tool is None:
            return f"工具执行出错: 没有名为 {call.name!r} 的工具"
        try:
            arguments = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"工具执行出错: 参数不是合法 JSON: {exc}"
        if not isinstance(arguments, dict):
            return "工具执行出错: 参数必须是一个 JSON 对象"

        # 工具本体是同步函数（读取文件、执行命令等会阻塞当前线程），
        # 如果直接在事件循环里调用，整个程序在工具执行期间都会卡住。
        # nanobot 里工具本身就是异步的（await 到底层 I/O）；教学版不重写所有工具，
        # 用 asyncio.to_thread 把同步调用折中地扔进线程池，事件循环照常处理其他会话。
        return await asyncio.to_thread(tool.run, arguments)
