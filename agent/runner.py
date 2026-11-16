"""部件 6（编排层）：AgentRunner —— 引擎的 ReAct 循环（对应 nanobot/agent/runner.py）。

每一轮做的事：
    1) 把全部消息发给模型（async 调用，等它返回）；
    2) 把模型回复（可能含工具调用）追加进消息列表；
    3) 如果模型没有请求工具 → 这就是最终答案，结束；
    4) 如果请求了工具 → 逐个执行，把结果作为 role="tool" 的消息回灌，回到第 1 步。

与旧版（同步）相比的三处升级：
    - provider.chat 是 async 的，这里用 await 调用，不阻塞事件循环；
    - 普通工具仍是同步函数，用 asyncio.to_thread 扔进线程池执行；
    - 协程工具（如 delegate）直接 await；同一轮的多个 concurrency_safe
      工具会并发执行（见 _execute_calls）。

分工说明（对应 nanobot 的架构）：
    runner 只管"这一条会话的 ReAct 循环"——它不知道有总线、有渠道、有多个会话；
    那些"谁把消息送进来、回复发去哪"的编排工作在 loop.py 里完成。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..providers import LLMProvider
from ..providers.base import DeltaCallback, Message, ToolCall, Usage
from ..utils.tracing import observation, record_iteration, record_tool_call
from .request import current_request_context
from .tools import Tool, available_tools

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

_EVENT_RESULT_CHARS = 1_200


def _missing_content(finish_reason: str) -> str:
    """模型没有给出文本时返回可诊断的信息，而不是让界面显示一片空白。"""
    if finish_reason in ("content_filter", "refusal"):
        return f"（模型拒绝回答或被内容安全策略拦截，finish_reason={finish_reason}）"
    if finish_reason == "length":
        return "（模型输出达到长度上限，未返回完整内容；请缩短输入后重试）"
    return (
        "（模型没有返回文本内容，"
        f"finish_reason={finish_reason}；请检查端点与模型是否支持当前请求）"
    )


class AgentRunner:
    def __init__(self, provider: LLMProvider, max_iterations: int = 10):
        self.provider = provider
        self.max_iterations = max_iterations

    async def run(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        usage: Usage | None = None,
    ) -> str:
        """在 messages 上跑 ReAct 循环（会原地往 messages 里追加内容），返回最终文本。

        注意：入参是"某一条会话的历史快照"，run 结束后里面会多出
        assistant / tool 消息——调用方（AgentLoop）只挑选最终答案写回会话存储。

        ``tools`` 为 None 时使用全局注册表中当前可用的工具；子代理会传入一份
        过滤后的工具列表（例如去掉 delegate / spawn，避免无限递归）。

        ``on_delta`` 不为 None 时逐段输出模型文本。多轮工具调用之间补一个换行，
        避免上一轮的前言和下一轮的结论黏在同一行。

        ``on_event`` 接收工具开始/结束等结构化事件，供界面展示执行状态；
        ``usage`` 不为 None 时把每次模型调用的 token 用量累加进去。
        """
        tool_list = available_tools() if tools is None else list(tools)
        tool_map = {tool.name: tool for tool in tool_list}
        schemas = [tool.schema() for tool in tool_list]
        last_char = ""

        for index in range(self.max_iterations):
            record_iteration()
            with observation(
                f"react.iteration.{index}",
                metadata={"message_count": len(messages)},
            ) as iteration:
                iteration_text = False

                async def emit(delta: str) -> None:
                    nonlocal iteration_text, last_char
                    if not delta:
                        return
                    iteration_text = True
                    last_char = delta[-1]
                    assert on_delta is not None
                    await on_delta(delta)

                # 1) 调模型（await：请求期间事件循环可以去处理别的会话的消息）
                response = await self.provider.chat(
                    messages,
                    tools=schemas,
                    on_delta=emit if on_delta is not None else None,
                )
                if usage is not None and response.usage is not None:
                    usage.add(response.usage)

                # 端点不支持流式时 provider 会回退为一次性返回；这里统一成
                # "文本已经交给回调"的语义，上层不必区分两种模式。
                if (
                    on_delta is not None
                    and response.content
                    and not iteration_text
                ):
                    await emit(response.content)

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
                    answer = response.content or _missing_content(response.finish_reason)
                    if on_delta is not None and not iteration_text:
                        await emit(answer)
                    iteration.update(output=answer)
                    return answer

                # 3b) 判定二（对应 nanobot 的 LLMResponse.should_execute_tools）：
                #     finish_reason 属于 refusal / content_filter / error 时，即使响应里
                #     带着 tool_calls，也不能执行——那往往是网关/安全过滤"借模型之口"
                #     注入的调用（nanobot 源码注释指向 issue #3220 的网关注入 bug）。
                #     不看 finish_reason 就信工具调用字段，会被这类注入骗执行工具。
                if not response.should_execute_tools:
                    answer = response.content or _missing_content(response.finish_reason)
                    if on_delta is not None and not iteration_text:
                        await emit(answer)
                    iteration.update(output=answer)
                    return answer

                # 4) 有工具调用 → 执行并把结果按原顺序回灌
                results = await self._execute_calls(
                    response.tool_calls, tool_map, on_event
                )
                for call, result in zip(response.tool_calls, results):
                    messages.append(
                        Message(role="tool", content=result, tool_call_id=call.id)
                    )
                if on_delta is not None and iteration_text and last_char != "\n":
                    await emit("\n")
                iteration.update(output=f"已执行 {len(results)} 个工具调用")
                # 循环回到顶部：模型看到工具结果后继续推理

        answer = "（已达到最大工具轮次，仍未得到最终答案）"
        if on_delta is not None:
            if last_char and last_char != "\n":
                await emit("\n")
            await emit(answer)
        return answer

    async def _execute_calls(
        self,
        calls: list[ToolCall],
        tool_map: dict[str, Tool],
        on_event: EventCallback | None,
    ) -> list[str]:
        """执行一轮里的全部工具调用。

        只有全部工具都明确标记为 ``concurrency_safe`` 时才并发；文件写入、
        执行命令这类有副作用的工具仍按模型给出的顺序串行，避免竞态。
        """
        if len(calls) > 1 and all(
            (tool := tool_map.get(call.name)) is not None and tool.concurrency_safe
            for call in calls
        ):
            return list(
                await asyncio.gather(
                    *(
                        self._execute(call, tool_map, on_event)
                        for call in calls
                    )
                )
            )
        results: list[str] = []
        for call in calls:
            results.append(await self._execute(call, tool_map, on_event))
        return results

    async def _execute(
        self,
        call: ToolCall,
        tool_map: dict[str, Tool],
        on_event: EventCallback | None,
    ) -> str:
        """执行单次工具调用：找工具 → 解析参数 → 审批 → 运行。"""
        started = time.perf_counter()
        tool = tool_map.get(call.name)
        side_effect = bool(tool is not None and tool.requires_approval)
        await self._emit(
            on_event,
            {
                "type": "tool_start",
                "id": call.id,
                "name": call.name,
                "arguments": _safe_json(call.arguments),
                "side_effect": side_effect,
            },
        )

        if tool is None:
            message = f"工具执行出错: 没有名为 {call.name!r} 的工具"
            record_tool_call(
                name=call.name,
                arguments=call.arguments,
                result=message,
                ok=False,
                duration_ms=0.0,
            )
            await self._finish_event(
                on_event, call, False, message, started, side_effect=side_effect
            )
            return message
        try:
            arguments = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError as exc:
            message = f"工具执行出错: 参数不是合法 JSON: {exc}"
            record_tool_call(
                name=call.name,
                arguments=call.arguments,
                result=message,
                ok=False,
                duration_ms=0.0,
            )
            await self._finish_event(
                on_event, call, False, message, started, side_effect=side_effect
            )
            return message
        if not isinstance(arguments, dict):
            message = "工具执行出错: 参数必须是一个 JSON 对象"
            record_tool_call(
                name=call.name,
                arguments=call.arguments,
                result=message,
                ok=False,
                duration_ms=0.0,
            )
            await self._finish_event(
                on_event, call, False, message, started, side_effect=side_effect
            )
            return message

        if not await self._approved(tool, arguments):
            message = (
                "工具调用未执行：该操作可能修改文件、执行命令或删除数据，"
                "但没有获得用户确认。请向用户说明将要执行的操作，"
                "等用户明确同意后再调用。"
            )
            record_tool_call(
                name=call.name,
                arguments=arguments,
                result=message,
                ok=False,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            await self._finish_event(
                on_event,
                call,
                False,
                message,
                started,
                denied=True,
                side_effect=side_effect,
            )
            return message

        # Tool.arun_with_status 内部区分同步/协程：同步工具走 asyncio.to_thread，
        # 协程工具留在事件循环里 await。
        ok, rendered = await tool.arun_with_status(arguments)
        await self._finish_event(
            on_event, call, ok, rendered, started, side_effect=side_effect
        )
        return rendered

    @staticmethod
    async def _approved(tool: Tool, arguments: dict) -> bool:
        """有副作用的工具必须得到明确批准；没有确认通道时按拒绝处理。"""
        from .tools.policy import approval_required

        if not approval_required(tool.requires_approval):
            return True
        context = current_request_context()
        callback = context.approval if context is not None else None
        if callback is None:
            return False
        try:
            return bool(await callback(tool.name, arguments))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("工具 %s 的确认流程失败，按拒绝处理", tool.name)
            return False

    @staticmethod
    async def _emit(on_event: EventCallback | None, event: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            await on_event(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("工具事件投递失败")

    async def _finish_event(
        self,
        on_event: EventCallback | None,
        call: ToolCall,
        ok: bool,
        result: str,
        started: float,
        *,
        denied: bool = False,
        side_effect: bool = False,
    ) -> None:
        preview = result
        if len(preview) > _EVENT_RESULT_CHARS:
            preview = preview[:_EVENT_RESULT_CHARS] + "…（已截断）"
        await self._emit(
            on_event,
            {
                "type": "tool_end",
                "id": call.id,
                "name": call.name,
                "ok": ok,
                "denied": denied,
                "side_effect": side_effect,
                "result": preview,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {"_raw": raw[:_EVENT_RESULT_CHARS]}
