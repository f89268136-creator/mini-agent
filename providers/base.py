"""部件 1：数据结构与模型接口（对应 nanobot 的 providers/base.py）。

- 数据结构：Message / ToolCall / LLMResponse / Usage——把"在程序里流转的数据"
  用 dataclass 固定下来，避免到处使用裸 dict、字段写错了也没人发现；
- LLMProvider：所有对话模型的统一抽象接口，引擎只依赖它、不关心底层是哪家模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """模型要求我们调用的一次工具。

    id:        这次调用的唯一编号（回灌结果时要带上，模型靠它配对）
    name:      工具名，对应工具注册表里的名字
    arguments: 模型给出的参数，是一段 JSON 字符串（此时还没解析成 dict）
    """

    id: str
    name: str
    arguments: str


@dataclass
class Message:
    """一条对话消息，role 决定它是谁说的。

    role 取值：
        "system"    系统设定（人设 / 规则），通常只在最前面放一条
        "user"      用户输入
        "assistant" 模型回复；如果它想调用工具，会同时带上 tool_calls
        "tool"      工具执行结果，必须带 tool_call_id，指明是哪次调用的结果
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        """转成 OpenAI /chat/completions 接口要求的 JSON 结构。"""
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Usage:
    """一次模型调用消耗的 token 数量（对应 nanobot providers/base.py 里 LLMUsage 的简化版）。

    为什么单独建一个类型而不是裸 dict：用量统计/计费代码到处读它，
    固定字段名比 d["prompt_tokens"] 这种写法少一半笔误。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """一次模型返回的规整结果（把厂商原始 JSON 解析成我们自己的对象）。"""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    # 兼容端点（本地模型、各种网关）不一定返回用量，缺省 None，统计代码要容忍。
    usage: Usage | None = None

    @property
    def has_tool_calls(self) -> bool:
        """判定算法：模型这一轮是不是在请求工具。

        这就是 nanobot 里 LLMResponse.has_tool_calls 的简化版——
        只要解析出了非空的 tool_calls，就说明该去执行工具，
        而不是把 content 直接当成最终答案。
        """
        return bool(self.tool_calls)

    @property
    def should_execute_tools(self) -> bool:
        """更严格的判定：有 tool_calls 之外，还要要求 finish_reason 正常。

        为什么不能只看 has_tool_calls：nanobot 真实踩过这个坑——某些网关在
        refusal / error 等异常状态下也会往响应里塞 tool_calls，照单执行会拿
        模型拒答/报错时的"半成品"参数去动用户的文件。所以再要求
        finish_reason 属于 ("tool_calls", "function_call", "stop") 这三个正常取值。
        """
        return self.has_tool_calls and self.finish_reason in (
            "tool_calls",
            "function_call",
            "stop",
        )


class LLMProvider(ABC):
    """所有模型提供者的统一接口。换模型 = 换一个实现，引擎代码一行都不用改。

    注意 chat 是 async 的：调用方用 `await provider.chat(...)`。
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        ...
