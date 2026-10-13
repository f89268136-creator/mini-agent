"""离线自测（异步版）：不连真实模型、不联网，验证 ReAct 主回路、工具执行守卫
（should_execute_tools）与 RetryingProvider 的重试策略。

运行方式（Windows GBK 控制台建议加 PYTHONIOENCODING=utf-8 前缀）：

    python -m mini_agent.tests.test_offline

覆盖五个场景：
    1) ReAct 主回路：模型先要求调用 get_current_time，回灌工具结果后给出最终答案；
    2) should_execute_tools 守卫：finish_reason 异常（content_filter / length）时，
       响应里即使带着 tool_calls，runner 也不能执行工具；
    3) 瞬时错误重试：前两次抛 RuntimeError("HTTP 500 ...")（类名不命中关键词，
       靠消息里的 500 命中瞬时判定），第三次成功；
    4) 非瞬时错误：ValueError("bad request") 立即向上抛，不浪费重试次数；
    5) finish_reason == "error" 的响应按失败处理，下一轮重试成功。
"""

from __future__ import annotations

import asyncio

from ..agent import Agent
from ..agent.tools import REGISTRY
from ..providers import LLMProvider, LLMResponse, Message, RetryingProvider, ToolCall


class FakeProvider(LLMProvider):
    """剧本：第 1 次要求调用 get_current_time，第 2 次校验工具回灌后给出最终答案。"""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
                finish_reason="tool_calls",
            )
        # 校验：工具结果确实被回灌，且配对编号正确
        last = messages[-1]
        assert last.role == "tool", "工具结果没有以 role=tool 回灌"
        assert last.tool_call_id == "call_1", "tool_call_id 配对错误"
        return LLMResponse(content=f"现在是 {last.content}", finish_reason="stop")


class GuardProvider(LLMProvider):
    """剧本：返回带 tool_calls 但 finish_reason 异常的响应（网关异常状态模拟）。"""

    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=f"（finish_reason={self.finish_reason}，本轮不执行工具）",
            tool_calls=[ToolCall(id="call_x", name="get_current_time", arguments="{}")],
            finish_reason=self.finish_reason,
        )


class FlakyProvider(LLMProvider):
    """剧本：前 failures 次抛瞬时错误（消息里的 500 命中瞬时判定），之后成功。"""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("HTTP 500 服务端错误（模拟瞬时故障）")
        return LLMResponse(content="重试后成功", finish_reason="stop")


class BadRequestProvider(LLMProvider):
    """剧本：每次都抛非瞬时错误（ValueError，类名与消息都不含可重试关键词）。"""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        raise ValueError("bad request: 参数不合法")


class ErrorThenOkProvider(LLMProvider):
    """剧本：第 1 次用 finish_reason="error" 的响应报告失败，第 2 次正常返回。"""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="网关内部错误", finish_reason="error")
        return LLMResponse(content="第二次成功", finish_reason="stop")


async def test_react_loop() -> None:
    """场景 1：ReAct 主回路（工具调用 -> 回灌 role=tool -> 最终答案）。"""
    agent = Agent(FakeProvider())
    answer = await agent.chat("现在几点？")

    assert "get_current_time" in REGISTRY, "内置工具未注册"
    assert "现在是" in answer, f"最终回答不符合预期: {answer}"

    roles = [m.role for m in agent.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"], (
        f"历史结构不符（应为 system/user/assistant/tool/assistant）: {roles}"
    )
    assert any(m.role == "tool" for m in agent.history), "工具结果没有进入历史"

    print("[1] ReAct 主回路：通过")
    print("    最终回答:", answer)
    print("    历史消息条数:", len(agent.history))
    for m in agent.history:
        if m.tool_calls:
            print("      -", m.role, f"<tool_calls={[c.name for c in m.tool_calls]}>")
        else:
            print("      -", m.role, (m.content or "")[:40])


async def test_should_execute_tools_guard() -> None:
    """场景 2：finish_reason 异常时，带 tool_calls 也不能执行工具。"""
    for reason in ("content_filter", "length"):
        provider = GuardProvider(reason)
        agent = Agent(provider)
        answer = await agent.chat("现在几点？")

        assert provider.calls == 1, f"守卫场景只应问模型一次，实际 {provider.calls} 次"
        tool_msgs = [m for m in agent.history if m.role == "tool"]
        assert not tool_msgs, (
            f"finish_reason={reason} 时不应执行工具，却回灌了 {len(tool_msgs)} 条 tool 消息"
        )
        expected = f"（finish_reason={reason}，本轮不执行工具）"
        assert answer == expected, f"应原样返回 content: {answer}"

    print("[2] should_execute_tools 守卫：通过（content_filter / length 均未执行工具）")


async def test_retrying_transient() -> None:
    """场景 3：瞬时错误（消息含 500）按指数退避重试后成功。"""
    inner = FlakyProvider(failures=2)
    # base_delay 调小到 0.01，让重试等待只有毫秒级，测试保持秒级完成
    provider = RetryingProvider(inner, max_attempts=3, base_delay=0.01)

    response = await provider.chat([Message(role="user", content="你好")])

    assert response.content == "重试后成功", f"重试结果不符: {response.content}"
    assert inner.calls == 3, f"应共调用 3 次（2 次失败 + 1 次成功），实际 {inner.calls}"

    print(f"[3] 瞬时错误重试：通过（共调用 {inner.calls} 次后成功）")


async def test_retrying_non_transient() -> None:
    """场景 4：非瞬时错误立即上抛，只调用 1 次。"""
    inner = BadRequestProvider()
    provider = RetryingProvider(inner, max_attempts=3, base_delay=0.01)

    try:
        await provider.chat([Message(role="user", content="你好")])
    except ValueError as exc:
        assert "bad request" in str(exc), f"异常内容不符: {exc}"
    else:
        raise AssertionError("非瞬时错误本应向上抛出，却正常返回了")

    assert inner.calls == 1, f"非瞬时错误不应重试，实际调用 {inner.calls} 次"

    print(f"[4] 非瞬时错误立即上抛：通过（调用 {inner.calls} 次后抛出 ValueError）")


async def test_retrying_error_response() -> None:
    """场景 5：finish_reason == "error" 的响应按失败处理并重试。"""
    inner = ErrorThenOkProvider()
    provider = RetryingProvider(inner, max_attempts=3, base_delay=0.01)

    response = await provider.chat([Message(role="user", content="你好")])

    assert response.finish_reason == "stop", f"最终应为正常响应: {response.finish_reason}"
    assert response.content == "第二次成功", f"重试结果不符: {response.content}"
    assert inner.calls == 2, f"应正好调用 2 次（1 次 error 响应 + 1 次成功），实际 {inner.calls}"

    print(f"[5] finish_reason=error 重试：通过（调用 {inner.calls} 次后拿到正常响应）")


async def _amain() -> None:
    await test_react_loop()
    await test_should_execute_tools_guard()
    await test_retrying_transient()
    await test_retrying_non_transient()
    await test_retrying_error_response()
    print("test_offline 离线自测通过（ReAct 回路 / 工具守卫 / 重试策略，全程离线）")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
