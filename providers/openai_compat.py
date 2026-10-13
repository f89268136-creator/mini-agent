"""部件 2：模型提供者——OpenAI 兼容实现（对应 nanobot/providers/）。

- OpenAICompatProvider 用官方 openai SDK 的 AsyncOpenAI 调用 chat.completions，
  网络等待期间不会卡住事件循环，可接入 OpenAI / DashScope / Ollama / 各种网关；
- RetryingProvider 是装饰器式包装器：指数退避重试，把"重试策略"从"发请求"里拆出来；
- LLMProvider 抽象基类与数据结构放在 base.py（对应 nanobot/providers/base.py）。

分工（为什么这样拆）：
    OpenAICompatProvider 只管"发请求 + 解析"，SDK 抛出的异常一律不吞、向上抛；
    是否重试、退避多久由外层 RetryingProvider 统一决定。
    这样换一个底层实现（比如 Anthropic）时，重试逻辑一行都不用重写。
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse, Message, ToolCall, Usage


class OpenAICompatProvider(LLMProvider):
    """接入任何"OpenAI 兼容"端点。"""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.model = model
        # 异步版 SDK 客户端：鉴权头、JSON 序列化、超时、连接复用仍由 SDK 统一处理，
        # 区别只是请求以 await 方式发出，不阻塞事件循环里的其他任务。
        # 本地模型常不校验密钥，SDK 又要求 key 非空，因此缺省填 "EMPTY"。
        self.client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        # 1) 组装参数并 await 发起请求。
        #    这里刻意不写 try/except：超时、限流、连接失败由 SDK 抛什么就向上抛什么，
        #    重试决策交给外层 RetryingProvider——每层只做一件事，好读也好测。
        request_kwargs: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            request_kwargs["tools"] = tools
        raw = await self.client.chat.completions.create(**request_kwargs)

        # 2) 把 SDK 对象解析成我们自己的 LLMResponse
        choice = raw.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (message.tool_calls or [])
        ]

        # 3) usage：部分兼容端点（本地模型、轻量网关）根本不返回用量，
        #    所以缺失时保持 None 而不是伪造 0，避免统计代码把"没数据"当"没消耗"。
        usage = None
        if raw.usage is not None:
            usage = Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
            )
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )


# "瞬时错误"判定的关键词（教学版，故意做得很粗）：
#   类名里带 timeout / connection / ratelimit 的异常（对应 openai SDK 的
#   APITimeoutError、APIConnectionError、RateLimitError），
#   或消息里出现常见的可重试 HTTP 状态码。
_TRANSIENT_NAME_HINTS = ("timeout", "connection", "ratelimit")
_TRANSIENT_TEXT_HINTS = ("429", "500", "502", "503", "504")


def _is_transient_error(exc: BaseException) -> bool:
    """粗略判断异常"等一等再试"有没有意义。

    nanobot 生产版在 providers/base.py 里有完整的错误分类表（状态码 408/409/429、
    timeout/connection 种类，甚至 429 里还要再区分"配额耗尽"与"真限流"哪些不该重试）；
    教学版只留最小规则：认不出是瞬时错误的，一律不重试，立刻抛。
    """
    name = type(exc).__name__.lower()
    if any(hint in name for hint in _TRANSIENT_NAME_HINTS):
        return True
    return any(hint in str(exc) for hint in _TRANSIENT_TEXT_HINTS)


class RetryingProvider(LLMProvider):
    """装饰器式包装器：包住任意 LLMProvider，失败时指数退避重试。

    用它包一层即可获得重试能力，被包的 provider 不用知道自己会被重试。

    两类失败都算"这一轮没成功"：
        1) inner.chat 抛异常——只有瞬时错误（见 _is_transient_error）才重试；
        2) inner.chat 不抛异常但返回 finish_reason == "error" 的响应——
           有些网关把错误塞进正常响应里，也按失败处理（在剩余次数内重试）。
    两条路径共用同一份重试预算 max_attempts；次数用尽后，异常就抛出、
    error 响应则原样返回给上层，由 runner 决定怎么展示。
    """

    def __init__(self, inner: LLMProvider, max_attempts: int = 3, base_delay: float = 0.5):
        self.inner = inner
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        for attempt in range(1, self.max_attempts + 1):
            is_last = attempt >= self.max_attempts
            try:
                response = await self.inner.chat(messages, tools)
            except Exception as exc:
                # 非瞬时错误（鉴权失败、参数错误等）重试也是白等；次数用尽同样抛。
                if is_last or not _is_transient_error(exc):
                    raise
            else:
                # inner 用返回值"报告失败"（finish_reason == "error"）视同异常处理：
                # 还有次数就重试；最后一次则把这份错误响应交回上层，不再吞掉。
                if response.finish_reason != "error" or is_last:
                    return response

            # 走到这里说明本轮失败且还有重试机会：
            # 指数退避——第 n 次重试前等 base_delay * 2^(n-1) 秒，给服务端喘息时间。
            # 用 asyncio.sleep（而不是 time.sleep），等待期间事件循环还能干别的活。
            await asyncio.sleep(self.base_delay * 2 ** (attempt - 1))

        raise RuntimeError("重试循环不应到达这里")  # pragma: no cover


def provider_from_env() -> LLMProvider:
    """从环境变量构造带重试的 provider（对应 nanobot 读取 config.json + 环境变量）。

    MODEL           模型名（默认 gpt-4o-mini）
    OPENAI_API_KEY  密钥（本地模型可留空）
    OPENAI_BASE_URL 端点（默认 OpenAI 官方；Ollama 用 http://localhost:11434/v1）
    """
    return RetryingProvider(
        OpenAICompatProvider(
            model=os.environ.get("MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    )


def build_provider(config) -> LLMProvider:
    """从配置对象构造带重试的 provider（对应 nanobot providers/factory.py）。

    这里刻意不 import config 模块，而是 duck-typing 读它的属性：
    核心层不需要认识装配层的类型，谁有这几个属性都能传进来，
    依赖方向保持"装配层 -> 核心层"单向。
    """
    inner = OpenAICompatProvider(
        model=getattr(config, "model", "gpt-4o-mini"),
        api_key=getattr(config, "api_key", None),
        api_base=getattr(config, "api_base", None) or "https://api.openai.com/v1",
        timeout=getattr(config, "timeout", None) or 60.0,
    )
    return RetryingProvider(
        inner,
        max_attempts=getattr(config, "retry_max_attempts", None) or 3,
        base_delay=getattr(config, "retry_base_delay", None) or 0.5,
    )
