"""部件：嵌入模型（Embedding），RAG 的基础（对应 nanobot providers 里的 embeddings）。

- Embedder 是异步抽象基类：把一段段文本变成一个个向量（list[float]）；
- OpenAICompatEmbedder 用官方 openai SDK 的 AsyncOpenAI 调用 OpenAI 兼容的
  embeddings，可接入 OpenAI（text-embedding-3-small）或 DashScope（text-embedding-v3）等。

同步改异步的动机：embed 是一次网络请求，旧版同步实现会把整个事件循环卡住；
改成 coroutine 后，RAG 摄入/检索与对话请求可以在同一个事件循环里并发。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod

from openai import AsyncOpenAI


class Embedder(ABC):
    """所有嵌入模型的统一接口。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """输入若干文本，返回等长的向量列表。"""

    async def aclose(self) -> None:
        """释放底层客户端资源；默认实现无需清理。"""
        return None

    def fingerprint(self) -> str:
        """标识"向量空间"来源；索引加载时用它判断旧向量是否还能复用。"""
        model = getattr(self, "model", "") or ""
        return f"{type(self).__name__}:{model}" if model else type(self).__name__


class OpenAICompatEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.model = model
        self._api_key = api_key or "EMPTY"
        self._api_base = api_base
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embed 的输入必须全部是字符串")
        # 每次调用创建客户端：RAG 工具在 asyncio.to_thread 的独立事件循环里
        # 用 asyncio.run 调用 embed，httpx 连接池不能跨事件循环共享。
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
            timeout=self._timeout,
            max_retries=2,
        )
        try:
            raw = await client.embeddings.create(model=self.model, input=texts)
        finally:
            await client.close()
        # 按 index 排序，保证第 i 个向量对应第 i 段输入文本
        data = sorted(raw.data, key=lambda item: item.index)
        vectors = [item.embedding for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"嵌入结果数量不匹配：输入 {len(texts)} 段，返回 {len(vectors)} 个向量"
            )
        return vectors

    async def aclose(self) -> None:
        return None

    def fingerprint(self) -> str:
        # 同一模型在不同端点（OpenAI / DashScope / 自建代理）上的向量空间不同，
        # 因此把端点也纳入指纹，避免跨端点复用旧向量。
        return f"{super().fingerprint()}@{self._api_base}"


class HashingEmbedder(Embedder):
    """离线确定性嵌入：特征哈希（feature hashing），不依赖网络与密钥。

    用词元 + 中文字符二元组做哈希，把每个词元映射到固定维度的某个分量并带符号累加，
    最后做 L2 归一化。它的语义能力远不如真实嵌入模型，但胜在完全离线、结果可复现，
    适合测试、无密钥环境和嵌入服务不可用时的降级检索。
    """

    _WORD_RE = re.compile(r"[a-z0-9_]+")
    _CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

    def __init__(self, dims: int = 512):
        if dims < 8:
            raise ValueError("HashingEmbedder 的 dims 必须 >= 8")
        self.dims = dims
        self.model = f"hashing-{dims}"

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        lowered = text.lower()
        tokens = cls._WORD_RE.findall(lowered)
        cjk = cls._CJK_RE.findall(lowered)
        tokens.extend(cjk)
        tokens.extend(cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1))
        return tokens

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embed 的输入必须全部是字符串")
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dims
            # 同一段文本里重复词元的贡献会叠加，等价于 TF 的简化版。
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dims
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    async def aclose(self) -> None:
        return None


def _default_embedding_model(api_base: str) -> str:
    # DashScope 与 OpenAI 官方的默认嵌入模型名不同，按端点给一个合理默认
    if "dashscope" in api_base:
        return "text-embedding-v3"
    return "text-embedding-3-small"


def _backend_from_env() -> str:
    backend = (
        os.environ.get("MINI_EMBEDDING_BACKEND")
        or os.environ.get("EMBEDDING_BACKEND")
        or "openai"
    )
    return backend.strip().lower()


def embedder_from_env(model: str | None = None) -> Embedder:
    """从环境变量构造嵌入模型。

    EMBEDDING_MODEL  嵌入模型名（不填则按端点给默认）
    EMBEDDING_BACKEND openai（默认）或 hashing（离线确定性，无需密钥）
    OPENAI_API_KEY / OPENAI_BASE_URL 与对话模型共用同一套环境变量
    """
    if _backend_from_env() == "hashing":
        return HashingEmbedder()
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAICompatEmbedder(
        model=model or os.environ.get("EMBEDDING_MODEL") or _default_embedding_model(api_base),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=api_base,
    )


def build_embedder(config) -> Embedder:
    """从配置对象构造嵌入模型（与对话 provider 共用端点与密钥）。"""
    backend = (getattr(config, "embedding_backend", None) or "openai").strip().lower()
    if backend == "hashing":
        return HashingEmbedder()
    api_base = getattr(config, "api_base", None) or "https://api.openai.com/v1"
    return OpenAICompatEmbedder(
        model=getattr(config, "embedding_model", None) or _default_embedding_model(api_base),
        api_key=getattr(config, "api_key", None),
        api_base=api_base,
        timeout=getattr(config, "timeout", None) or 60.0,
    )
