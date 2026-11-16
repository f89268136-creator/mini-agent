"""工具功能包（对应 nanobot/agent/tools/）。

    registry.py   注册表：Tool 数据结构 + @register_tool 装饰器 + schema 自动生成
    builtin.py    9 个基础工具：时间 / 计算器 / 读写文件 / 编辑历史 / 列目录 / 查找 / 搜索 / 执行命令
    knowledge.py  RAG：语义切块 + 持久化索引 + 混合检索（ingest/search/list/delete/stats）
    memory.py     长期记忆：memory_save / search / list / forget
    todo.py       任务清单：todo_write / todo_read
    multi_agent.py 多智能体：delegate（同步等待）/ spawn（后台派生）
    web.py        网络：web_search（必应）+ fetch_webpage（正文抓取）

导入本包 = 导入下面 6 个功能模块：其中的函数在导入时被 @register_tool
注册进 REGISTRY，模型因此能看到全部工具。
"""

from .policy import (
    configure_multi_agent,
    configure_policy,
    describe_policy,
    multi_agent_enabled,
    resolve_workspace_path,
)
from .registry import (
    REGISTRY,
    Tool,
    all_tool_schemas,
    available_tools,
    configure_tool_timeout,
    register_tool,
)

# 逐个子模块导入（导入副作用：把各自的工具注册进 REGISTRY）。
# 顺序决定模型看到的工具顺序：基础 -> RAG -> 记忆 -> 多智能体 -> 网络。
from . import builtin  # noqa: F401
from . import knowledge  # noqa: F401
from . import memory  # noqa: F401
from . import todo  # noqa: F401
from . import multi_agent  # noqa: F401
from . import web  # noqa: F401
from .knowledge import (
    KnowledgeBase,
    SearchHit,
    configure_knowledge_base,
    reset_knowledge_base,
)

__all__ = [
    "REGISTRY",
    "Tool",
    "all_tool_schemas",
    "available_tools",
    "configure_tool_timeout",
    "register_tool",
    "KnowledgeBase",
    "SearchHit",
    "configure_knowledge_base",
    "reset_knowledge_base",
    "configure_policy",
    "configure_multi_agent",
    "describe_policy",
    "multi_agent_enabled",
    "resolve_workspace_path",
]
