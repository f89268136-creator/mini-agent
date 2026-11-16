"""把底层异常翻译成用户能照做的提示。

模型调用失败时，直接把 ``openai.AuthenticationError`` 的原始文案抛给终端，
用户往往只看到一长串 JSON 和堆栈，不知道是密钥、网络还是模型名的问题。
这里按异常类型给出"原因 + 下一步动作"，同时保留截断后的原始信息供排查。
"""

from __future__ import annotations

import asyncio
import socket

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

_MAX_DETAIL_CHARS = 240


def _short(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > _MAX_DETAIL_CHARS:
        text = text[:_MAX_DETAIL_CHARS] + "…"
    return text or type(exc).__name__


def _connection_detail(exc: BaseException) -> str:
    cause = exc.__cause__ or exc.__context__
    if isinstance(cause, socket.gaierror):
        return "域名解析失败"
    if isinstance(cause, ConnectionRefusedError):
        return "连接被拒绝，服务可能未启动或端口不对"
    if isinstance(cause, (TimeoutError, asyncio.TimeoutError)):
        return "建立连接超时"
    return "网络不可达或 TLS 握手失败"


def format_exception(exc: BaseException) -> str:
    """返回一句可操作的中文说明（不包含密钥，原始信息会被截断）。"""
    if isinstance(exc, (asyncio.TimeoutError, APITimeoutError)):
        return (
            "请求超时：模型服务在规定时间内没有响应。"
            "请检查网络/代理，或调大配置项 timeout。"
        )
    if isinstance(exc, AuthenticationError):
        return (
            "鉴权失败（HTTP 401）：OPENAI_API_KEY 未设置、无效，"
            "或与当前 api_base 不匹配。"
        )
    if isinstance(exc, PermissionDeniedError):
        return (
            "无权访问（HTTP 403）：当前密钥没有该模型/端点的权限，"
            "或账号被限流/封禁。"
        )
    if isinstance(exc, NotFoundError):
        return (
            "端点或模型不存在（HTTP 404）：请检查 api_base 是否包含 /v1，"
            "以及 model 是否为该端点的有效模型名。"
        )
    if isinstance(exc, RateLimitError):
        return "请求被限流（HTTP 429）：请稍后重试，并检查账号配额与并发量。"
    if isinstance(exc, BadRequestError):
        return (
            "请求参数被拒绝（HTTP 400）：请检查 model 名称、上下文长度，"
            f"以及端点是否支持 tools/function calling。原始信息：{_short(exc)}"
        )
    if isinstance(exc, APIConnectionError):
        return (
            f"无法连接模型服务（{_connection_detail(exc)}）："
            "请检查 api_base、网络/代理，以及本地模型是否已启动。"
        )
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", "?")
        return f"模型服务返回 HTTP {status}：{_short(exc)}"
    if isinstance(exc, socket.gaierror):
        return "域名解析失败：请检查 api_base 的主机名与 DNS/代理设置。"
    if isinstance(exc, ConnectionRefusedError):
        return "连接被拒绝：目标服务未启动或端口不对，请检查 api_base。"
    if isinstance(exc, (ConnectionError, OSError)):
        return f"网络错误：{_short(exc)}"
    return f"{type(exc).__name__}: {_short(exc)}"
