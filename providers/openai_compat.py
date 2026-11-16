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
import logging
import os
import random

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from ..utils.netutil import is_loopback_url
from .base import DeltaCallback, LLMProvider, LLMResponse, Message, ToolCall, Usage

logger = logging.getLogger(__name__)
_MAX_BACKOFF_SECONDS = 30.0
# 这些状态码通常意味着端点不认 stream 参数；在还没输出任何文本前回退一次非流式。
_STREAM_FALLBACK_STATUS = frozenset({400, 404, 405, 415, 422})


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
        # max_retries=0：关闭 SDK 内建重试，避免与 RetryingProvider 叠加成
        # "3 × 3" 次请求；重试策略只在最外层实现一次。
        self.client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
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

        if on_delta is None:
            return await self._chat_once(request_kwargs)

        emitted = False

        async def emit(delta: str) -> None:
            nonlocal emitted
            if not delta:
                return
            emitted = True
            await on_delta(delta)

        try:
            return await self._chat_stream(request_kwargs, emit)
        except APIStatusError as exc:
            # 不是所有兼容端点都支持 stream。还没输出任何内容时回退一次；
            # 已经输出过就不能再发第二遍，否则用户会看到重复文本。
            if emitted or exc.status_code not in _STREAM_FALLBACK_STATUS:
                raise
            logger.warning(
                "端点不支持流式输出（HTTP %s），回退为非流式: %s",
                exc.status_code,
                self.model,
            )
            response = await self._chat_once(request_kwargs)
            if response.content:
                await emit(response.content)
            return response

    async def _chat_once(self, request_kwargs: dict) -> LLMResponse:
        raw = await self.client.chat.completions.create(**request_kwargs)
        return self._parse_response(raw)

    async def _chat_stream(
        self, request_kwargs: dict, on_delta: DeltaCallback
    ) -> LLMResponse:
        stream = await self.client.chat.completions.create(
            stream=True, **request_kwargs
        )
        content_parts: list[str] = []
        tool_slots: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: Usage | None = None
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = Usage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                        await on_delta(text)
                    for call in getattr(delta, "tool_calls", None) or []:
                        slot = tool_slots.setdefault(
                            call.index if call.index is not None else len(tool_slots),
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if call.id:
                            slot["id"] = call.id
                        function = getattr(call, "function", None)
                        name = getattr(function, "name", None) if function else None
                        if name:
                            # 标准实现只在首块给完整名字；少数网关会分片。
                            if not slot["name"] or (
                                name != slot["name"] and not slot["name"].endswith(name)
                            ):
                                slot["name"] += name
                        arguments = (
                            getattr(function, "arguments", None) if function else None
                        )
                        if arguments:
                            slot["arguments"] += arguments
                reason = getattr(choice, "finish_reason", None)
                if reason:
                    finish_reason = reason
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # 关闭失败不影响已经拿到的内容
                    logger.debug("关闭流式响应失败", exc_info=True)

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=slot["arguments"],
            )
            for index, slot in sorted(tool_slots.items())
            if slot["name"]
        ]
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _parse_response(raw) -> LLMResponse:
        # 2) 把 SDK 对象解析成我们自己的 LLMResponse
        if not raw.choices:
            raise RuntimeError("模型返回了空 choices 列表")
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

    async def aclose(self) -> None:
        await self.client.close()


# 非 openai SDK 异常（自建网关、测试替身）的兜底分类关键词
_TRANSIENT_NAME_HINTS = ("timeout", "connection", "ratelimit", "temporar", "overloaded")
_TRANSIENT_TEXT_HINTS = ("408", "409", "429", "500", "502", "503", "504")


def _is_transient_error(exc: BaseException) -> bool:
    """判断"等一等再试"是否有意义。

    openai SDK 异常按类型/HTTP 状态码精确分类；其他异常按类名与文案兜底。
    未识别为瞬时的错误（鉴权失败、参数错误等）不重试，立即上抛。
    """
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (408, 409, 429) or 500 <= exc.status_code < 600
    name = type(exc).__name__.lower()
    if any(hint in name for hint in _TRANSIENT_NAME_HINTS):
        return True
    return any(hint in str(exc) for hint in _TRANSIENT_TEXT_HINTS)


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """读取服务端 Retry-After 头（秒），并限制在合理上限内。"""
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(seconds, _MAX_BACKOFF_SECONDS))


class RetryingProvider(LLMProvider):
    """装饰器式包装器：包住任意 LLMProvider，失败时指数退避重试。

    两类失败都算"这一轮没成功"：
        1) inner.chat 抛异常——只有瞬时错误（见 _is_transient_error）才重试；
        2) inner.chat 不抛异常但返回 finish_reason == "error" 的响应——
           有些网关把错误塞进正常响应里，也按失败处理。
    退避带 ±20% 抖动，避免大量客户端同时重试；服务端给出 Retry-After
    时以它为准，并统一限制在 30 秒内。两条路径共用 max_attempts 预算。
    """

    def __init__(self, inner: LLMProvider, max_attempts: int = 3, base_delay: float = 0.5):
        if max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")
        if base_delay < 0:
            raise ValueError("base_delay 必须 >= 0")
        self.inner = inner
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        for attempt in range(1, self.max_attempts + 1):
            is_last = attempt >= self.max_attempts
            emitted = False

            async def emit(delta: str) -> None:
                nonlocal emitted
                emitted = True
                assert on_delta is not None
                await on_delta(delta)

            try:
                response = await self.inner.chat(
                    messages,
                    tools,
                    on_delta=emit if on_delta is not None else None,
                )
            except Exception as exc:
                # 已经流式输出过就不能重试：再发一次会让用户看到重复文本。
                if is_last or emitted or not _is_transient_error(exc):
                    raise
                delay = self._delay(attempt, exc)
                logger.warning(
                    "LLM 请求失败（第 %d/%d 次尝试），%.2f 秒后重试: %s: %s",
                    attempt,
                    self.max_attempts,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
                continue

            # inner 用返回值"报告失败"（finish_reason == "error"）视同异常处理：
            # 还有次数就重试；最后一次（或已经输出过内容）则把响应交回上层。
            if response.finish_reason != "error" or is_last or emitted:
                return response
            delay = self._delay(attempt, None)
            logger.warning(
                "网关返回 finish_reason=error（第 %d/%d 次尝试），%.2f 秒后重试",
                attempt,
                self.max_attempts,
                delay,
            )
            await asyncio.sleep(delay)

        raise RuntimeError("重试循环不应到达这里")  # pragma: no cover

    def _delay(self, attempt: int, exc: BaseException | None) -> float:
        """指数退避 + 抖动 + 服务端 Retry-After + 上限。"""
        delay = min(self.base_delay * 2 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
        if delay > 0:
            delay *= random.uniform(0.8, 1.2)
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return max(0.0, delay)

    async def aclose(self) -> None:
        await self.inner.aclose()


def provider_from_env() -> LLMProvider:
    """从环境变量构造带重试的 provider（对应 nanobot 读取 config.json + 环境变量）。

    MODEL           模型名（默认 gpt-4o-mini）
    OPENAI_API_KEY  密钥（本地模型可留空）
    OPENAI_BASE_URL 端点（默认 OpenAI 官方；Ollama 用 http://localhost:11434/v1）

    远程端点缺少密钥时直接抛 ValueError，避免调用方在首次请求时才发现问题。
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key and not is_loopback_url(api_base):
        raise ValueError(
            "未配置 OPENAI_API_KEY：远程端点需要密钥；"
            "使用本地模型时请把 OPENAI_BASE_URL 指向 127.0.0.1/localhost"
        )
    return RetryingProvider(
        OpenAICompatProvider(
            model=os.environ.get("MODEL", "gpt-4o-mini"),
            api_key=api_key,
            api_base=api_base,
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
    retry_base_delay = getattr(config, "retry_base_delay", None)
    return RetryingProvider(
        inner,
        max_attempts=getattr(config, "retry_max_attempts", None) or 3,
        # 0 是合法值（立即重试），不能用 `or 0.5` 吞掉
        base_delay=0.5 if retry_base_delay is None else retry_base_delay,
    )
