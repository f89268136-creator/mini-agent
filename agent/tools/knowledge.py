"""部件：RAG 知识库（对应 nanobot 里"检索增强生成"相关能力）。

RAG 的思路：
    1) 摄入：把长文档切成小段（chunk），用嵌入模型把每段变成向量，存起来；
    2) 检索：用户提问时，把问题也变成向量，用余弦相似度找出最相关的几段；
    3) 生成：把这几段原文交给模型作为参考，模型据此回答（这一步由 runner 完成）。

这里用纯 Python 实现向量与余弦相似度，不依赖 numpy / 向量数据库，
适合理解 RAG 的最小原理；生产中可替换成真正的向量库。

异步边界说明：KnowledgeBase 的 ingest / search 内部要 await 嵌入模型，
所以它们是 async 方法；而两个知识库工具对外仍是普通同步函数——
工具运行环境是 runner 的 asyncio.to_thread 工作线程，线程里没有事件循环，
同步函数在末尾用 asyncio.run 驱动 async 方法正合适。
"""

from __future__ import annotations

import asyncio
import math
import os

from ...providers.embeddings import Embedder, embedder_from_env
from .registry import register_tool


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把长文本按字符数切成带重叠的小段。

    重叠（overlap）的作用：避免一句话/一个语义正好从切口处被切断，
    相邻片段共享一部分内容，检索更稳。
    """
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：衡量两个向量方向有多接近，范围约 -1~1，越接近 1 越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeBase:
    """一个极简的内存向量库：片段、向量、来源三者按下标一一对应。"""

    def __init__(self, embedder: Embedder, chunk_size: int = 500):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunks: list[str] = []
        self.vectors: list[list[float]] = []
        self.sources: list[str] = []

    async def ingest(self, text: str, source: str = "") -> int:
        """摄入一段文本：切块 -> await 批量嵌入 -> 存入。返回新增片段数。"""
        pieces = chunk_text(text, self.chunk_size)
        if not pieces:
            return 0
        vectors = await self.embedder.embed(pieces)
        for piece, vector in zip(pieces, vectors):
            self.chunks.append(piece)
            self.vectors.append(vector)
            self.sources.append(source)
        return len(pieces)

    async def search(self, query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
        """检索与问题最相关的片段，返回 (片段, 来源, 相似度) 列表，按相似度降序。"""
        if not self.vectors:
            return []
        query_vector = (await self.embedder.embed([query]))[0]
        scored = [
            (cosine_similarity(query_vector, vector), index)
            for index, vector in enumerate(self.vectors)
        ]
        scored.sort(reverse=True)
        return [
            (self.chunks[index], self.sources[index], score)
            for score, index in scored[:top_k]
        ]


# 知识库要在"先摄入、后检索"多次工具调用之间保持状态，因此用一个延迟初始化的单例。
_KB: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase(embedder_from_env())
    return _KB


# ----------------------------------------------------------------------
# RAG 工具：导入本模块时自动注册（对应 nanobot 的知识库检索能力）
# ----------------------------------------------------------------------
@register_tool
def knowledge_ingest(path: str) -> str:
    """把本地文本或 Markdown 文件加入知识库；之后可用 knowledge_search 检索其中内容。"""
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        text = f.read()
    kb = get_knowledge_base()
    # 工具函数本身跑在 to_thread 工作线程里（没有事件循环），
    # 所以能放心用 asyncio.run 把 async 方法驱动到底。
    count = asyncio.run(kb.ingest(text, source=path))
    return f"已从 {path} 摄入 {count} 个片段，知识库当前共 {len(kb.chunks)} 个片段。"


@register_tool
def knowledge_search(query: str, top_k: int = 3) -> str:
    """在已摄入的知识库中检索与问题最相关的原文片段（RAG 检索），用于回答前查阅资料。"""
    kb = get_knowledge_base()
    if not kb.chunks:
        return "知识库为空，请先用 knowledge_ingest 工具摄入文档。"
    hits = asyncio.run(kb.search(query, top_k=top_k))
    blocks = []
    for rank, (text, source, score) in enumerate(hits, start=1):
        blocks.append(
            f"[片段{rank} 来源:{source or '未知'} 相关度:{score:.2f}]\n{text}"
        )
    return "\n\n".join(blocks)
