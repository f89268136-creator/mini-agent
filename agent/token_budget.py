"""轻量 token 估算：不依赖具体分词器，给上下文预算与压缩触发提供一个稳定近似。

中文等 CJK 字符通常接近 1 字 1 token，英文/代码大约 4 字符 1 token。
估算只用于"提前压缩"的决策，宁可略高估，也不追求精确计费。
"""

from __future__ import annotations

import json
import re

from ..providers.base import Message

_CJK_RE = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)
# 每条消息的角色、分隔符等结构开销，按经验给一个小常数。
_MESSAGE_OVERHEAD = 4
_TOOL_CALL_OVERHEAD = 4

# 为模型输出预留的 token：预算只约束输入，不能把上下文窗口吃满。
OUTPUT_RESERVE_TOKENS = 1_024


def estimate_tokens(text: str | None) -> int:
    """估算一段文本的 token 数；空文本返回 0。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def estimate_message_tokens(message: Message) -> int:
    total = _MESSAGE_OVERHEAD + estimate_tokens(message.content)
    for call in message.tool_calls or []:
        total += (
            _TOOL_CALL_OVERHEAD
            + estimate_tokens(call.name)
            + estimate_tokens(call.arguments)
        )
    return total


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_tool_schemas_tokens(schemas: list[dict]) -> int:
    """估算 tools 参数的 token 开销；Schema 每轮都会发送，必须计入输入预算。"""
    if not schemas:
        return 0
    return estimate_tokens(json.dumps(schemas, ensure_ascii=False))
