"""部件：嵌入模型（Embedding），RAG 的基础（对应 nanobot providers 里的 embeddings）。

- Embedder 是异步抽象基类：把一段段文本变成一个个向量（list[float]）；
- OpenAICompatEmbedder 用官方 openai SDK 的 AsyncOpenAI 调用 OpenAI 兼容的
  embeddings，可接入 OpenAI（text-embedding-3-small）或 DashScope（text-embedding-v3）等。

同步改异步的动机：embed 是一次网络请求，旧版同步实现会把整个事件循环卡住；
改成 coroutine 后，RAG 摄入/检索与对话请求可以在同一个事件循环里并发。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from openai import AsyncOpenAI


class Embedder(ABC):
    """所有嵌入模型的统一接口。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """输入若干文本，返回等长的向量列表。"""


class OpenAICompatEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.model = model
        # 异步版 SDK 客户端：统一处理鉴权、超时与连接复用；本地模型缺省用 "EMPTY" 占位。
        self.client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            timeout=timeout,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raw = await self.client.embeddings.create(model=self.model, input=texts)
        # 按 index 排序，保证第 i 个向量对应第 i 段输入文本
        data = sorted(raw.data, key=lambda item: item.index)
        return [item.embedding for item in data]


def _default_embedding_model(api_base: str) -> str:
    # DashScope 与 OpenAI 官方的默认嵌入模型名不同，按端点给一个合理默认
    if "dashscope" in api_base:
        return "text-embedding-v3"
    return "text-embedding-3-small"


def embedder_from_env() -> Embedder:
    """从环境变量构造嵌入模型。

    EMBEDDING_MODEL  嵌入模型名（不填则按端点给默认）
    OPENAI_API_KEY / OPENAI_BASE_URL 与对话模型共用同一套环境变量
    """
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAICompatEmbedder(
        model=os.environ.get("EMBEDDING_MODEL") or _default_embedding_model(api_base),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=api_base,
    )
