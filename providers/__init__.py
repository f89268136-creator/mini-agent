"""模型功能包（对应 nanobot/providers/）。

三个文件的分工：
    base.py           数据结构（Message / ToolCall / LLMResponse / Usage）
                      + LLMProvider 抽象基类
    openai_compat.py  OpenAI 兼容实现（AsyncOpenAI）+ RetryingProvider 重试包装 + 工厂函数
    embeddings.py     嵌入模型抽象 + OpenAI 兼容实现（RAG 的基础）

导入本包即可拿到全部模型层类型：
    from mini_agent.providers import LLMProvider, Message, provider_from_env
"""

from .base import LLMProvider, LLMResponse, Message, ToolCall, Usage
from .embeddings import Embedder, OpenAICompatEmbedder, embedder_from_env
from .openai_compat import (
    OpenAICompatProvider,
    RetryingProvider,
    build_provider,
    provider_from_env,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "Usage",
    "Embedder",
    "OpenAICompatEmbedder",
    "embedder_from_env",
    "OpenAICompatProvider",
    "RetryingProvider",
    "build_provider",
    "provider_from_env",
]
