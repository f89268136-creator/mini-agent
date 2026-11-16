"""工具注册表（对应 nanobot/agent/tools/registry.py 的最小版）。

一个工具 = 一个普通 Python 函数 + 名字 + 描述 + 参数 schema：
    - 用 @register_tool 装饰函数即可注册（存进模块级字典 REGISTRY）；
    - 参数的 JSON Schema 从函数的类型注解自动生成；
    - 模型看到这些 schema 后，才知道"有哪些工具可用、参数怎么填"。

为什么大多数工具坚持同步（Tool.run 也是同步的）：工具里可能干重活（读文件、
跑命令、网络请求），在 async 环境里同步阻塞会卡住事件循环。处理办法不是把每个
工具都改 async，而是由 runner 用 asyncio.to_thread 把同步函数丢到工作线程执行，
核心层保持"工具是普通函数"的简单心智模型（nanobot 生产版同样在线程/沙箱里跑工具）。

少数工具必须留在事件循环里（例如 delegate 要 await 子代理、spawn 要创建后台任务），
它们写成 ``async def``，注册表会自动标记 ``is_async``，runner 用 ``Tool.arun``
直接 await；同一轮里带 ``concurrency_safe`` 标记的工具还可以并发执行。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Callable

from ...utils.tracing import observation, record_tool_call
from .policy import shell_enabled

# Python 类型 -> JSON Schema 类型名
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# 注解字符串 -> 类型：因为本文件开了 `from __future__ import annotations`，
# 运行时读到的类型注解是字符串（如 "float"），需要先查这张表还原成真实类型。
_TYPE_BY_NAME = {t.__name__: t for t in _TYPE_MAP}

# 单次工具输出上限：工具结果会回灌给模型并落盘，必须有硬上限，
# 否则一次超大输出就能挤爆上下文、拖慢后续所有轮次。
MAX_TOOL_OUTPUT_CHARS = 100_000

# 单个同步工具的执行上限（秒），0 表示不限；MiniBot 启动时按配置覆盖。
# 注意：同步工具跑在工作线程里，超时只能放弃等待，无法真正终止线程。
_TOOL_TIMEOUT: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict
    # 可选可用性判断：配置层面关闭某类工具时，连 schema 也不发给模型。
    available: Callable[[], bool] | None = None
    # 协程工具（如 delegate / spawn）必须在事件循环里 await，不能丢进线程池。
    is_async: bool = False
    # 多个此类工具出现在同一轮时，runner 会并发执行而不是逐个等待。
    concurrency_safe: bool = False
    # 有副作用的工具（写文件、执行命令、删除知识库来源）：执行前需要用户确认。
    requires_approval: bool = False

    def is_available(self) -> bool:
        return self.available() if self.available is not None else True

    @staticmethod
    def _render(result: object) -> str:
        text = str(result)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            return (
                text[:MAX_TOOL_OUTPUT_CHARS]
                + f"\n…（输出已截断，原长 {len(text)} 字符）"
            )
        return text

    @staticmethod
    def _error(exc: Exception) -> str:
        return f"工具执行出错: {type(exc).__name__}: {exc}"

    def run(self, arguments: dict) -> str:
        """同步执行工具（供非异步调用方使用）。

        任何异常都转成文字返回给模型，让模型自己纠错，
        而不是让异常直接把整个程序带崩。
        """
        if self.is_async:
            return "工具执行出错: 异步工具必须由 AgentRunner 执行"
        try:
            return self._render(self.func(**arguments))
        except Exception as exc:  # 这里就是要兜住所有错误，反馈给模型
            return self._error(exc)

    async def arun_with_status(self, arguments: dict) -> tuple[bool, str]:
        """异步执行工具：协程函数直接 await，同步函数放进线程池。

        返回 ``(是否成功, 渲染后的结果)``，让 runner 能把执行状态同步给界面；
        工具异常仍会转换成文字回灌模型，而不是向上抛出。
        """
        with observation(
            f"tool.{self.name}", as_type="tool", input=arguments
        ) as span:
            started = time.perf_counter()
            try:
                if self.is_async:
                    # 协程工具（delegate/spawn 等）由各自的超时与限流管理。
                    result = await self.func(**arguments)
                else:
                    result = await self._await_sync_tool(arguments)
                rendered = self._render(result)
                span.update(output=rendered)
                record_tool_call(
                    name=self.name,
                    arguments=arguments,
                    result=rendered,
                    ok=True,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                return True, rendered
            except Exception as exc:  # CancelledError 属于 BaseException，不会被吞掉
                message = self._error(exc)
                span.update(output=message, level="ERROR")
                record_tool_call(
                    name=self.name,
                    arguments=arguments,
                    result=message,
                    ok=False,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                return False, message

    async def _await_sync_tool(self, arguments: dict):
        """在线程池里执行同步工具，并施加单工具 deadline。"""
        call = asyncio.to_thread(self.func, **arguments)
        timeout = _TOOL_TIMEOUT
        if timeout <= 0:
            return await call
        try:
            return await asyncio.wait_for(call, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"工具执行超时（超过 {timeout:g} 秒）；"
                "同步工具可能仍在后台线程中运行，副作用未必已停止"
            ) from exc

    def schema(self) -> dict:
        """OpenAI tools 字段要求的结构。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# 全局注册表：工具名 -> Tool
REGISTRY: dict[str, Tool] = {}


def configure_tool_timeout(seconds: float) -> None:
    """设置单个同步工具的执行上限（秒）；0 表示不限。"""
    global _TOOL_TIMEOUT
    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("tool_timeout 必须 >= 0")
    _TOOL_TIMEOUT = seconds


def register_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    available: Callable[[], bool] | None = None,
    concurrency_safe: bool = False,
    requires_approval: bool = False,
    parameters: dict | None = None,
):
    """装饰器：把一个函数注册成工具。

    两种写法都支持：
        @register_tool
        def now() -> str: ...

        @register_tool(name="calc", description="计算器")
        def calculate(expression: str) -> str: ...

    ``async def`` 工具会被自动识别，由 runner 在事件循环里 await；
    ``concurrency_safe=True`` 表示同一轮的多个此类调用可以并发执行；
    ``parameters`` 用于覆盖自动生成的 schema（数组、嵌套对象等复杂参数）。
    """

    def wrap(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "").strip()
        REGISTRY[tool_name] = Tool(
            name=tool_name,
            description=tool_desc,
            func=fn,
            parameters=parameters or _build_parameters(fn),
            available=available,
            is_async=inspect.iscoroutinefunction(fn),
            concurrency_safe=concurrency_safe,
            requires_approval=requires_approval,
        )
        return fn

    # 支持 @register_tool 不带括号直接使用
    if func is not None:
        return wrap(func)
    return wrap


def _build_parameters(func: Callable) -> dict:
    """读取函数签名和类型注解，自动生成 parameters JSON Schema。"""
    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            annotation = str
        elif isinstance(annotation, str):
            # 字符串注解（上面 _TYPE_BY_NAME 解释过）还原成类型；认不出的按 string 处理
            annotation = _TYPE_BY_NAME.get(annotation, str)
        properties[pname] = {"type": _TYPE_MAP.get(annotation, "string")}
        # 没有默认值的参数视为"必填"
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def available_tools() -> list[Tool]:
    """当前策略下允许暴露给模型的工具（顺序与注册顺序一致）。"""
    return [
        tool
        for tool in REGISTRY.values()
        if tool.is_available()
        and (tool.name != "run_shell" or shell_enabled())
    ]


def all_tool_schemas() -> list[dict]:
    """返回给模型看的工具 schema；被策略禁用的工具不会出现在列表里。"""
    return [tool.schema() for tool in available_tools()]
