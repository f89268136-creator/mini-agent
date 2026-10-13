"""RAG 离线自测（异步版）：不连网络，用 FakeEmbedder 验证切块、向量检索，
以及 KnowledgeBase 的异步接口（await kb.ingest / await kb.search）。

运行方式（Windows GBK 控制台建议加 PYTHONIOENCODING=utf-8 前缀）：

    python -m mini_agent.tests.test_rag_offline
"""

from __future__ import annotations

import asyncio

from ..agent.tools.knowledge import KnowledgeBase, chunk_text
from ..providers.embeddings import Embedder


class FakeEmbedder(Embedder):
    """把文本映射成二维"关键词向量"：含 apple 落在第 1 维，含 banana 落在第 2 维。

    注意 embed 是 async 的：KnowledgeBase 的 ingest / search 内部会 await 它，
    这样嵌入（真实场景是网络请求）等待期间不会卡住事件循环。
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "apple" in lowered else 0.0,
                    1.0 if "banana" in lowered else 0.0,
                ]
            )
        return vectors


async def _amain() -> None:
    # 1) 切块：足够长的文本应切成多段（相邻片段带重叠）
    long_text = "苹果。" * 300
    chunks = chunk_text(long_text, chunk_size=200)
    assert len(chunks) >= 2, "长文本没有被切成多段"
    assert all(len(c) <= 200 for c in chunks), "存在超过 chunk_size 的片段"

    # 2) 摄入 + 检索：全程 await（embedder 是异步的）
    kb = KnowledgeBase(FakeEmbedder())
    added_apple = await kb.ingest("Apple is a sweet fruit.", source="fruits.txt")
    assert added_apple == 1, f"短文本应摄入 1 个片段，实际 {added_apple}"
    await kb.ingest("Banana is yellow and long.", source="fruits.txt")

    hits = await kb.search("tell me about apple", top_k=1)
    assert hits, "没有检索到结果"
    top_text, top_source, score = hits[0]
    assert "Apple" in top_text, f"检索到的不是最相关片段: {top_text}"
    assert top_source == "fruits.txt", f"来源标注错误: {top_source}"
    assert score == 1.0, f"完全匹配的相似度应为 1.0，实际 {score}"

    # 3) 空库检索应安全返回空列表（不报错、不查嵌入）
    empty = KnowledgeBase(FakeEmbedder())
    assert await empty.search("anything") == [], "空库检索应返回空列表"

    # 4) ingest 返回值 = 新增片段数，且与 len(kb.chunks) 对得上
    kb2 = KnowledgeBase(FakeEmbedder(), chunk_size=200)
    expected = chunk_text(long_text, chunk_size=200)
    count = await kb2.ingest(long_text, source="long.txt")
    assert count == len(expected) == len(kb2.chunks), (
        f"ingest 返回值与片段数不一致: count={count}, "
        f"expected={len(expected)}, chunks={len(kb2.chunks)}"
    )

    print("RAG 离线自测通过")
    print("切块数量:", len(chunks))
    print("命中片段:", top_text, "来源:", top_source, "相似度:", score)
    print("长文本 ingest 新增片段数:", count)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
