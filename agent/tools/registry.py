"""工具注册表（对应 nanobot/agent/tools/registry.py 的最小版）。

一个工具 = 一个普通 Python 函数 + 名字 + 描述 + 参数 schema：
    - 用 @register_tool 装饰函数即可注册（存进模块级字典 REGISTRY）；
    - 参数的 JSON Schema 从函数的类型注解自动生成；
    - 模型看到这些 schema 后，才知道"有哪些工具可用、参数怎么填"。

为什么工具函数坚持同步（Tool.run 也是同步的）：工具里可能干重活（读文件、
跑命令、网络请求），在 async 环境里同步阻塞会卡住事件循环。处理办法不是把每个
工具都改 async，而是由 runner 用 asyncio.to_thread 把 Tool.run 丢到工作线程执行，
核心层保持"工具是普通函数"的简单心智模型（nanobot 生产版同样在线程/沙箱里跑工具）。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

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


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict

    def run(self, arguments: dict) -> str:
        """执行工具。

        任何异常都转成文字返回给模型，让模型自己纠错，
        而不是让异常直接把整个程序带崩。
        """
        try:
            result = self.func(**arguments)
            return str(result)
        except Exception as exc:  # 这里就是要兜住所有错误，反馈给模型
            return f"工具执行出错: {type(exc).__name__}: {exc}"

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


def register_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
):
    """装饰器：把一个函数注册成工具。

    两种写法都支持：
        @register_tool
        def now() -> str: ...

        @register_tool(name="calc", description="计算器")
        def calculate(expression: str) -> str: ...
    """

    def wrap(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "").strip()
        REGISTRY[tool_name] = Tool(
            name=tool_name,
            description=tool_desc,
            func=fn,
            parameters=_build_parameters(fn),
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


def all_tool_schemas() -> list[dict]:
    """返回给模型看的全部工具 schema。"""
    return [tool.schema() for tool in REGISTRY.values()]


def get_tool(name: str) -> Tool | None:
    return REGISTRY.get(name)
