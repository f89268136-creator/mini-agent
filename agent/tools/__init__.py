"""工具功能包（对应 nanobot/agent/tools/）。

    registry.py   注册表：Tool 数据结构 + @register_tool 装饰器 + schema 自动生成
    builtin.py    7 个基础工具：时间 / 计算器 / 读写文件 / 列目录 / 执行命令
    knowledge.py  RAG：切块 + 纯 Python 向量库 + knowledge_ingest / knowledge_search
    web.py        网络：web_search（必应）+ fetch_webpage（正文抓取）

导入本包 = 导入下面 4 个模块：builtin / knowledge / web 里的函数在导入时
被 @register_tool 注册进 REGISTRY，模型因此能看到全部工具。
"""

from .registry import REGISTRY, Tool, all_tool_schemas, get_tool, register_tool

# 逐个子模块导入（导入副作用：把各自的工具注册进 REGISTRY）
from . import builtin  # noqa: F401
from . import knowledge  # noqa: F401
from . import web  # noqa: F401

__all__ = ["REGISTRY", "Tool", "all_tool_schemas", "get_tool", "register_tool"]
