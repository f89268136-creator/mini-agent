"""RAG 知识库：语义切块、持久化索引与混合检索。

相比早期"全内存 + 定长切块 + 纯余弦"的实现，这一版补齐了可落地的检索工程链路：

- **切块**：先按 Markdown 标题 / 空行切成段落，再按句子装箱；超长段落硬切，
  相邻片段保留尾部重叠，避免一句话被切断后语义丢失。
- **持久化**：片段与向量写入 ``index.jsonl``（首行元数据 + 每行一个片段），
  采用临时文件 + ``os.replace`` 原子替换；进程重启后无需重新摄入。
- **嵌入指纹**：索引记录嵌入模型与端点；换了嵌入模型/端点后旧向量自动作废，
  首次检索时按当前模型重建，避免"维度相同但向量空间不同"的静默错配。
- **混合检索**：BM25 关键词分 + 余弦向量分加权融合，兼顾专有名词与语义相似；
  嵌入服务不可用时自动退化为纯关键词检索。
- **MMR 重排**：在相关性之外惩罚与已选片段过于相似的候选，减少重复段落。
- **索引管理**：按来源删除、目录批量摄入、来源/片段统计。

线程模型：工具是同步函数，由 runner 丢进工作线程，因此这里用 ``threading.RLock``
保护内存结构；``asyncio.to_thread`` 只用于把原子写盘移出事件循环。一个进程内的
多个工具线程可以安全并发，跨进程同时写同一个目录不在支持范围内。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import struct
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ...providers.embeddings import Embedder, embedder_from_env
from .policy import resolve_workspace_path, workspace_root
from .registry import register_tool

logger = logging.getLogger(__name__)

_MAX_INGEST_CHARS = 2_000_000
_MAX_TOP_K = 20
_MAX_DIR_FILES = 200
_INDEX_FILENAME = "index.jsonl"
_INDEX_VERSION = 2

_TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".log",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".go",
        ".rs",
        ".sql",
        ".html",
        ".css",
    }
)

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+")
_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(slots=True)
class SearchHit:
    """一条检索结果；行号来自原始文档（1 开始，闭区间）。"""

    text: str
    source: str
    score: float
    heading: str = ""
    position: int = 0
    start_line: int = 0
    end_line: int = 0
    chunk_id: str = ""


@dataclass(slots=True)
class _Chunk:
    id: str
    source: str
    text: str
    heading: str = ""
    position: int = 0
    start_line: int = 0
    end_line: int = 0
    ingested_at: str = ""
    vector: list[float] = field(default_factory=list)


# ----------------------------------------------------------------------
# 切块
# ----------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _semantic_blocks(text: str) -> list[tuple[str, str, int, int]]:
    """按 Markdown 标题与空行切成语义块，返回 (最近标题, 块文本, 起始行, 结束行)。"""
    blocks: list[tuple[str, str, int, int]] = []
    heading = ""
    buffer: list[str] = []
    buffer_start = 0
    buffer_end = 0
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            blocks.append((heading, body, buffer_start, buffer_end))
        buffer = []

    for line_no, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if not buffer:
                buffer_start = line_no
            buffer.append(raw.rstrip())
            buffer_end = line_no
            continue
        if not in_fence and _HEADING_RE.match(stripped):
            flush()
            heading = stripped.lstrip("#").strip()
            buffer_start = line_no
            buffer.append(stripped)
            buffer_end = line_no
            continue
        if not in_fence and not stripped:
            flush()
            continue
        if not buffer:
            buffer_start = line_no
        buffer.append(raw.rstrip())
        buffer_end = line_no
    flush()

    # Markdown 常写成"标题 + 空行 + 正文"，flush 会产生只有标题的块。
    # 把它并入下一块，避免标题单独成片或只出现在上一片段的尾部。
    merged: list[tuple[str, str, int, int]] = []
    pending = ""
    pending_start = 0
    pending_end = 0
    for heading, body, start_line, end_line in blocks:
        if _HEADING_RE.match(body) and "\n" not in body:
            pending = f"{pending}\n{body}" if pending else body
            if not pending_start:
                pending_start = start_line
            pending_end = end_line
            continue
        if pending:
            # 补回标题与正文之间被分隔掉的空行，保证块内换行数与原始行号一致。
            gap = max(0, start_line - pending_end - 1)
            body = f"{pending}{chr(10) * (gap + 1)}{body}"
            start_line = pending_start
            pending = ""
            pending_start = 0
            pending_end = 0
        merged.append((heading, body, start_line, end_line))
    if pending:
        merged.append((heading, pending, pending_start, pending_end))
    return merged


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """按句子边界切分，并保留每句在原文中的字符偏移。"""
    spans: list[tuple[str, int, int]] = []
    start = 0
    for match in _SENTENCE_RE.finditer(text):
        end = match.start()
        if end > start:
            spans.append((text[start:end], start, end))
        start = match.end()
    if start < len(text):
        spans.append((text[start:], start, len(text)))
    return spans


def _split_long(block: str, limit: int) -> list[tuple[str, int, int]]:
    """把超长段落按句子装箱；返回 (文本, 起始偏移, 结束偏移)。"""
    if len(block) <= limit:
        return [(block, 0, len(block))]

    units: list[tuple[str, int, int]] = []
    current: tuple[str, int, int] | None = None

    for segment, seg_start, seg_end in _sentence_spans(block):
        # 去掉首尾空白，同时保持偏移仍指向原文
        seg_start += len(segment) - len(segment.lstrip())
        seg_end -= len(segment) - len(segment.rstrip())
        text = segment.strip()
        if not text:
            continue

        while len(text) > limit:
            cut = text.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            part = text[:cut].rstrip()
            if part:
                if current is not None:
                    units.append(current)
                    current = None
                units.append((part, seg_start, seg_start + len(part)))
            remainder = text[cut:]
            seg_start += cut + (len(remainder) - len(remainder.lstrip()))
            text = remainder.strip()

        if not text:
            continue
        if current is None:
            current = (text, seg_start, seg_end)
        elif len(current[0]) + 1 + len(text) <= limit:
            current = (current[0] + "\n" + text, current[1], seg_end)
        else:
            units.append(current)
            current = (text, seg_start, seg_end)
    if current is not None:
        units.append(current)
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    """取上一片段的尾部作为下一片段的重叠前缀，尽量从句子/换行边界开始。"""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    for marker in ("\n\n", "\n", "。", "！", "？", "；", ". ", "; "):
        index = tail.find(marker)
        if index != -1:
            return tail[index + len(marker) :].strip()
    return tail.strip()


def _build_chunks(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[str, str, int, int, int]]:
    """返回 (片段文本, 标题, 序号, 起始行, 结束行) 列表。"""
    normalized = _normalize_text(text)
    if not normalized.strip():
        return []
    overlap = max(0, min(overlap, chunk_size - 1))

    units: list[tuple[str, str, int, int]] = []
    for heading, body, start_line, _end_line in _semantic_blocks(normalized):
        for piece, offset_start, offset_end in _split_long(body, chunk_size):
            piece_start = start_line + body.count("\n", 0, offset_start)
            piece_end = start_line + body.count("\n", 0, offset_end)
            units.append((piece, heading, piece_start, piece_end))

    chunks: list[tuple[str, str, int, int, int]] = []
    current = ""
    current_heading = ""
    current_start = 0
    current_end = 0
    position = 0
    for piece, heading, start_line, end_line in units:
        candidate = piece if not current else current + "\n\n" + piece
        if len(candidate) <= chunk_size:
            if not current:
                current_heading = heading
                current_start = start_line
            current = candidate
            current_end = end_line
            continue

        if current:
            chunks.append(
                (current, current_heading, position, current_start, current_end)
            )
            position += 1
            tail = _overlap_tail(current, overlap)
            combined = f"{tail}\n\n{piece}" if tail else piece
            if len(combined) <= chunk_size:
                current = combined
            else:
                current = piece
            current_start = start_line
            current_end = end_line
        else:
            # _split_long 已保证 piece <= chunk_size；这里只作防御。
            while len(piece) > chunk_size:
                chunks.append(
                    (piece[:chunk_size], heading, position, start_line, start_line)
                )
                position += 1
                piece = piece[chunk_size:]
            current = piece
            current_start = start_line
            current_end = end_line
        current_heading = heading
    if current:
        chunks.append((current, current_heading, position, current_start, current_end))
    return chunks


# ----------------------------------------------------------------------
# 相似度与词法打分
# ----------------------------------------------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：衡量两个向量方向有多接近，范围约 -1~1。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    """英文按单词、中文按单字 + 二元组切词，无需第三方分词器。"""
    lowered = text.lower()
    tokens = _WORD_RE.findall(lowered)
    cjk = _CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1))
    return tokens


def _bm25_scores(
    query_tokens: list[str],
    counters: list[Counter[str]],
    lengths: list[int],
) -> dict[int, float]:
    """轻量 BM25：只看查询词，文档频率在本次检索时现算。"""
    unique = list(dict.fromkeys(query_tokens))
    if not unique or not counters:
        return {}

    total = len(counters)
    average_length = sum(lengths) / total or 1.0
    document_frequency = {
        token: sum(1 for counter in counters if token in counter) for token in unique
    }
    k1, b = 1.5, 0.75

    scores: dict[int, float] = {}
    for index, counter in enumerate(counters):
        length = lengths[index] or 1
        score = 0.0
        for token in unique:
            frequency = counter.get(token, 0)
            if not frequency:
                continue
            idf = math.log(
                1.0 + (total - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            score += idf * (
                frequency * (k1 + 1)
            ) / (frequency + k1 * (1 - b + b * length / average_length))
        if score > 0:
            scores[index] = score

    if scores:
        maximum = max(scores.values())
        if maximum > 0:
            scores = {index: value / maximum for index, value in scores.items()}
    return scores


def _mmr_select(
    candidates: list[tuple[int, float]],
    vectors: list[list[float]],
    limit: int,
    lambda_: float,
) -> list[int]:
    """Maximal Marginal Relevance：相关性高且彼此不重复的片段优先。"""
    remaining = sorted(candidates, key=lambda item: item[1], reverse=True)
    selected: list[int] = []
    while remaining and len(selected) < limit:
        if not selected:
            index, _score = remaining.pop(0)
        else:
            best_position = 0
            best_value = float("-inf")
            for position, (index, relevance) in enumerate(remaining):
                redundancy = 0.0
                vector = vectors[index] if index < len(vectors) else []
                if vector:
                    for chosen in selected:
                        chosen_vector = vectors[chosen] if chosen < len(vectors) else []
                        if chosen_vector:
                            redundancy = max(
                                redundancy, cosine_similarity(vector, chosen_vector)
                            )
                value = lambda_ * relevance - (1.0 - lambda_) * redundancy
                if value > best_value:
                    best_value = value
                    best_position = position
            index, _score = remaining.pop(best_position)
        selected.append(index)
    return selected


# ----------------------------------------------------------------------
# 索引持久化辅助
# ----------------------------------------------------------------------
def _encode_vector(vector: list[float]) -> str:
    """float32 + base64：比 JSON 数字数组小一半以上，精度足够做余弦检索。"""
    payload = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(payload).decode("ascii")


def _decode_vector(payload: str) -> list[float]:
    raw = base64.b64decode(payload.encode("ascii"))
    if len(raw) % 4:
        raise ValueError("向量长度不是 4 的倍数")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeBase:
    """线程安全的片段索引：切块、嵌入、持久化、混合检索。"""

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 500,
        *,
        overlap: int = 50,
        index_dir: str | os.PathLike[str] | None = None,
        min_score: float = 0.0,
        lexical_weight: float = 0.35,
        mmr_lambda: float = 0.7,
        default_top_k: int = 4,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size 必须 >= 1")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 必须在 0..chunk_size-1 之间")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score 必须在 0..1 之间")
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight 必须在 0..1 之间")
        if not 0 <= mmr_lambda <= 1:
            raise ValueError("mmr_lambda 必须在 0..1 之间")
        if default_top_k < 1:
            raise ValueError("default_top_k 必须 >= 1")

        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_score = float(min_score)
        self.lexical_weight = float(lexical_weight)
        self.mmr_lambda = float(mmr_lambda)
        self.default_top_k = int(default_top_k)

        self.index_path = (
            Path(index_dir).expanduser() / _INDEX_FILENAME if index_dir else None
        )
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._chunks: list[_Chunk] = []
        self._counters: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._dims: int | None = None
        self._needs_reindex = False
        self.last_search_degraded = False
        self.last_persist_error: str | None = None

        self._load()

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    # ------------------------------------------------------------------
    # 摄入与写入
    # ------------------------------------------------------------------
    async def ingest(
        self, text: str, source: str = "", *, replace_source: bool = False
    ) -> int:
        """切块 -> 批量嵌入 -> 写入索引。

        ``replace_source=True`` 时先移除同一来源的旧片段（重新摄入同一文件时使用）；
        默认追加，便于把 source 当作分类标签。完全相同的片段在同一来源内只保留一份。
        """
        source = (source or "").strip()
        pieces = _build_chunks(text, self.chunk_size, self.overlap)
        if not pieces:
            if replace_source and source:
                await self.delete_source(source)
            return 0

        seen: set[str] = set()
        selected: list[tuple[str, str, int, int, int]] = []
        for piece, heading, position, start_line, end_line in pieces:
            digest = hashlib.sha1(piece.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            selected.append((piece, heading, position, start_line, end_line))

        vectors = await self.embedder.embed([piece for piece, *_rest in selected])
        if len(vectors) != len(selected):
            raise RuntimeError(
                f"嵌入结果数量不匹配：输入 {len(selected)} 段，返回 {len(vectors)} 个向量"
            )

        dims = {len(vector) for vector in vectors if vector}
        if len(dims) > 1:
            raise RuntimeError("同一次嵌入返回了不同维度的向量，请检查嵌入服务")
        new_dims = dims.pop() if dims else 0

        with self._lock:
            if self._dims and new_dims and new_dims != self._dims:
                raise RuntimeError(
                    f"嵌入维度不一致：知识库为 {self._dims} 维，本次为 {new_dims} 维。"
                    f"请删除索引文件 {self.index_path}，或清空知识库后重新摄入。"
                )
            if replace_source and source:
                self._remove_source_locked(source)

            now = _now_iso()
            base_position = len(self._chunks)
            for offset, ((piece, heading, _position, start_line, end_line), vector) in enumerate(
                zip(selected, vectors)
            ):
                chunk = _Chunk(
                    id=hashlib.sha1(
                        f"{source}|{base_position + offset}|{piece}".encode("utf-8")
                    ).hexdigest()[:16],
                    source=source,
                    text=piece,
                    heading=heading,
                    position=base_position + offset,
                    start_line=start_line,
                    end_line=end_line,
                    ingested_at=now,
                    vector=list(vector),
                )
                self._chunks.append(chunk)
                self._index_chunk_locked(chunk)
            if new_dims:
                self._dims = new_dims

        await self._persist()
        return len(selected)

    async def delete_source(self, source: str) -> int:
        """删除某个来源的全部片段，返回删除数量。"""
        source = (source or "").strip()
        with self._lock:
            removed = self._remove_source_locked(source)
        if removed:
            await self._persist()
        return removed

    async def clear(self) -> int:
        """清空全部片段（保留索引文件与元数据）。"""
        with self._lock:
            removed = len(self._chunks)
            self._chunks.clear()
            self._counters.clear()
            self._lengths.clear()
            self._dims = None
            self._needs_reindex = False
        await self._persist()
        return removed

    def _remove_source_locked(self, source: str) -> int:
        keep = [chunk for chunk in self._chunks if chunk.source != source]
        removed = len(self._chunks) - len(keep)
        if not removed:
            return 0
        self._chunks = keep
        self._rebuild_lexical_locked()
        self._dims = None
        for chunk in self._chunks:
            if chunk.vector:
                self._dims = len(chunk.vector)
                break
        return removed

    def _index_chunk_locked(self, chunk: _Chunk) -> None:
        counter = Counter(_tokenize(chunk.text))
        self._counters.append(counter)
        self._lengths.append(sum(counter.values()))

    def _rebuild_lexical_locked(self) -> None:
        self._counters = [Counter(_tokenize(chunk.text)) for chunk in self._chunks]
        self._lengths = [sum(counter.values()) for counter in self._counters]

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    async def search_hits(
        self, query: str, top_k: int | None = None
    ) -> list[SearchHit]:
        """混合检索 + MMR 重排，返回带标题/序号的检索结果。"""
        query = (query or "").strip()
        if not query:
            return []
        limit = self._clamp_top_k(self.default_top_k if top_k is None else top_k)
        self.last_search_degraded = False

        with self._lock:
            if not self._chunks:
                return []
            needs_reindex = self._needs_reindex or any(
                not chunk.vector for chunk in self._chunks
            )

        if needs_reindex:
            try:
                await self.ensure_vectors()
            except Exception:
                logger.warning("知识库向量重建失败，本次检索退化为关键词模式", exc_info=True)
                self.last_search_degraded = True

        with self._lock:
            texts = [chunk.text for chunk in self._chunks]
            sources = [chunk.source for chunk in self._chunks]
            headings = [chunk.heading for chunk in self._chunks]
            positions = [chunk.position for chunk in self._chunks]
            start_lines = [chunk.start_line for chunk in self._chunks]
            end_lines = [chunk.end_line for chunk in self._chunks]
            chunk_ids = [chunk.id for chunk in self._chunks]
            vectors = [list(chunk.vector) for chunk in self._chunks]
            counters = list(self._counters)
            lengths = list(self._lengths)

        query_vector: list[float] | None = None
        if self.lexical_weight < 1.0:
            try:
                embedded = await self.embedder.embed([query])
                if embedded:
                    query_vector = embedded[0]
                else:
                    self.last_search_degraded = True
            except Exception:
                logger.warning("查询嵌入失败，本次检索退化为关键词模式", exc_info=True)
                self.last_search_degraded = True

        # 运行期换嵌入器（维度随之变化）时，旧向量不能与新查询向量比较。
        if query_vector is not None:
            known_dims = {len(vector) for vector in vectors if vector}
            if known_dims and known_dims != {len(query_vector)}:
                query_vector = None
                self.last_search_degraded = True

        lexical_weight = self.lexical_weight
        if query_vector is None:
            lexical_weight = 1.0 if lexical_weight > 0 else 0.0
        vector_weight = 1.0 - lexical_weight

        lexical_scores = _bm25_scores(_tokenize(query), counters, lengths)
        candidates: list[tuple[int, float]] = []
        for index in range(len(texts)):
            vector_score = 0.0
            if query_vector is not None and vectors[index]:
                vector_score = max(
                    0.0, cosine_similarity(query_vector, vectors[index])
                )
            score = (
                lexical_weight * lexical_scores.get(index, 0.0)
                + vector_weight * vector_score
            )
            if score > 0 and score >= self.min_score:
                candidates.append((index, score))

        if not candidates:
            return []

        selected = _mmr_select(candidates, vectors, limit, self.mmr_lambda)
        scores = dict(candidates)
        return [
            SearchHit(
                text=texts[index],
                source=sources[index],
                heading=headings[index],
                position=positions[index],
                score=scores[index],
                start_line=start_lines[index],
                end_line=end_lines[index],
                chunk_id=chunk_ids[index],
            )
            for index in selected
        ]

    async def ensure_vectors(self) -> int:
        """为缺少向量的片段补算嵌入（索引迁移/换模型/降级恢复时使用）。"""
        with self._lock:
            pending = [
                (chunk.id, chunk.text) for chunk in self._chunks if not chunk.vector
            ]
        if not pending:
            with self._lock:
                self._needs_reindex = False
            return 0

        vectors = await self.embedder.embed([text for _id, text in pending])
        if len(vectors) != len(pending):
            raise RuntimeError(
                f"嵌入结果数量不匹配：输入 {len(pending)} 段，返回 {len(vectors)} 个向量"
            )

        dims = {len(vector) for vector in vectors if vector}
        if len(dims) > 1:
            raise RuntimeError("同一次嵌入返回了不同维度的向量，请检查嵌入服务")
        new_dims = dims.pop() if dims else 0

        by_id = {chunk_id: vector for (chunk_id, _text), vector in zip(pending, vectors)}
        with self._lock:
            if self._dims and new_dims and new_dims != self._dims:
                raise RuntimeError(
                    f"嵌入维度不一致：知识库为 {self._dims} 维，本次为 {new_dims} 维。"
                )
            for chunk in self._chunks:
                vector = by_id.get(chunk.id)
                if vector:
                    chunk.vector = list(vector)
            if new_dims:
                self._dims = new_dims
            self._needs_reindex = any(not chunk.vector for chunk in self._chunks)

        await self._persist()
        return len(vectors)

    async def reindex(self, *, force: bool = False) -> int:
        """重建向量；``force=True`` 时丢弃现有向量并按当前嵌入模型全部重算。"""
        with self._lock:
            if force:
                for chunk in self._chunks:
                    chunk.vector = []
                self._dims = None
            self._needs_reindex = any(not chunk.vector for chunk in self._chunks)
        return await self.ensure_vectors()

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------
    def list_sources(self) -> list[dict[str, object]]:
        """按来源聚合片段数、字符数与最近更新时间。"""
        with self._lock:
            groups: dict[str, dict[str, object]] = {}
            for chunk in self._chunks:
                key = chunk.source or "(未命名)"
                item = groups.setdefault(
                    key,
                    {
                        "source": key,
                        "chunks": 0,
                        "chars": 0,
                        "updated_at": "",
                    },
                )
                item["chunks"] = int(item["chunks"]) + 1
                item["chars"] = int(item["chars"]) + len(chunk.text)
                updated = str(item["updated_at"])
                if chunk.ingested_at > updated:
                    item["updated_at"] = chunk.ingested_at
        return sorted(groups.values(), key=lambda item: str(item["source"]))

    def stats(self) -> dict[str, object]:
        """索引概况；``needs_reindex`` 为真表示有片段缺向量、检索时会补算。"""
        with self._lock:
            chunk_count = len(self._chunks)
            char_count = sum(len(chunk.text) for chunk in self._chunks)
            source_count = len({chunk.source for chunk in self._chunks})
            missing = sum(1 for chunk in self._chunks if not chunk.vector)
        return {
            "sources": source_count,
            "chunks": chunk_count,
            "chars": char_count,
            "dims": self._dims,
            "embedder": self._fingerprint(),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "lexical_weight": self.lexical_weight,
            "mmr_lambda": self.mmr_lambda,
            "min_score": self.min_score,
            "index_path": str(self.index_path) if self.index_path else None,
            "missing_vectors": missing,
            "needs_reindex": missing > 0,
        }

    @staticmethod
    def _clamp_top_k(value: int) -> int:
        return max(1, min(int(value), _MAX_TOP_K))

    def _fingerprint(self) -> str:
        fingerprint = getattr(self.embedder, "fingerprint", None)
        if callable(fingerprint):
            return str(fingerprint())
        return type(self.embedder).__name__

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self) -> None:
        path = self.index_path
        if path is None or not path.exists():
            return

        meta: dict[str, object] | None = None
        records: list[dict[str, object]] = []
        corrupt = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        corrupt += 1
                        logger.warning("知识库索引第 %s 行不是合法 JSON，已跳过", line_number)
                        continue
                    if not isinstance(record, dict):
                        corrupt += 1
                        continue
                    if record.get("type") == "meta" and meta is None:
                        meta = record
                    elif record.get("type") == "chunk":
                        records.append(record)
        except (OSError, UnicodeDecodeError):
            logger.warning("无法读取知识库索引 %s", path, exc_info=True)
            return

        if corrupt:
            logger.warning("知识库索引 %s 有 %s 行损坏，已跳过", path, corrupt)

        version = 0
        try:
            version = int(meta.get("version", 0)) if meta else 0
        except (TypeError, ValueError):
            logger.warning("知识库索引版本字段损坏，按旧版本处理")
        if meta and version > _INDEX_VERSION:
            logger.warning(
                "知识库索引版本 %s 高于当前支持的 %s，仍尝试读取", version, _INDEX_VERSION
            )

        current_fingerprint = self._fingerprint()
        stored_fingerprint = str(meta.get("embedder", "")) if meta else ""
        stale_vectors = not meta or stored_fingerprint != current_fingerprint

        loaded: list[_Chunk] = []
        for record in records:
            try:
                text = str(record.get("text", ""))
                if not text:
                    continue
                vector: list[float] = []
                if not stale_vectors and record.get("vec"):
                    vector = _decode_vector(str(record["vec"]))
                loaded.append(
                    _Chunk(
                        id=str(record.get("id", "")),
                        source=str(record.get("source", "")),
                        text=text,
                        heading=str(record.get("heading", "")),
                        position=int(record.get("position", 0)),
                        start_line=int(record.get("start_line", 0) or 0),
                        end_line=int(record.get("end_line", 0) or 0),
                        ingested_at=str(record.get("ingested_at", "")),
                        vector=vector,
                    )
                )
            except (TypeError, ValueError):
                corrupt += 1

        with self._lock:
            self._chunks = loaded
            self._rebuild_lexical_locked()
            self._dims = None
            for chunk in self._chunks:
                if chunk.vector:
                    self._dims = len(chunk.vector)
                    break
            self._needs_reindex = any(not chunk.vector for chunk in self._chunks)

        if stale_vectors and loaded:
            logger.info(
                "知识库索引的嵌入指纹已变化（%s -> %s），将在下次检索时重建向量",
                stored_fingerprint or "无",
                current_fingerprint,
            )

    def _snapshot_lines_locked(self) -> list[str]:
        meta = {
            "type": "meta",
            "version": _INDEX_VERSION,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedder": self._fingerprint(),
            "dims": self._dims,
        }
        lines = [json.dumps(meta, ensure_ascii=False, separators=(",", ":"))]
        for chunk in self._chunks:
            record: dict[str, object] = {
                "type": "chunk",
                "id": chunk.id,
                "source": chunk.source,
                "heading": chunk.heading,
                "position": chunk.position,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "ingested_at": chunk.ingested_at,
            }
            if chunk.vector:
                record["vec"] = _encode_vector(chunk.vector)
            lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        return lines

    def _write_snapshot(self, lines: list[str]) -> None:
        path = self.index_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".index-", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.writelines(f"{line}\n" for line in lines)
                handle.flush()
                os.fsync(handle.fileno())
            with self._write_lock:
                os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    async def _persist(self) -> None:
        if self.index_path is None:
            return
        with self._lock:
            lines = self._snapshot_lines_locked()
        try:
            await asyncio.to_thread(self._write_snapshot, lines)
            self.last_persist_error = None
        except Exception as exc:  # 索引写失败不应让检索/摄入整体不可用
            self.last_persist_error = f"{type(exc).__name__}: {exc}"
            logger.exception("知识库索引写入失败: %s", self.index_path)


# ----------------------------------------------------------------------
# 进程级知识库：与 MemoryStore 一样，一次进程只服务一套装配。
# ----------------------------------------------------------------------
_KB: KnowledgeBase | None = None


def configure_knowledge_base(
    embedder: Embedder,
    chunk_size: int = 500,
    *,
    overlap: int = 50,
    index_dir: str | os.PathLike[str] | None = None,
    min_score: float = 0.0,
    lexical_weight: float = 0.35,
    mmr_lambda: float = 0.7,
    default_top_k: int = 4,
) -> KnowledgeBase:
    """替换进程级知识库（MiniBot 装配时调用，也可由 SDK 使用者自定义）。"""
    global _KB
    _KB = KnowledgeBase(
        embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        index_dir=index_dir,
        min_score=min_score,
        lexical_weight=lexical_weight,
        mmr_lambda=mmr_lambda,
        default_top_k=default_top_k,
    )
    return _KB


def reset_knowledge_base() -> None:
    """清空进程级知识库（测试或显式重建时使用）。"""
    global _KB
    _KB = None


def get_knowledge_base() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase(
            embedder_from_env(),
            index_dir=os.environ.get("MINI_KNOWLEDGE_DIR") or None,
        )
    return _KB


# ----------------------------------------------------------------------
# 引用收集：工具线程把命中追加到当前回合的桶里，AgentLoop 据此落盘引用。
# ----------------------------------------------------------------------
_CITATION_BUCKET: ContextVar[list[SearchHit] | None] = ContextVar(
    "knowledge_citation_bucket", default=None
)


@contextmanager
def citation_scope() -> Iterator[list[SearchHit]]:
    """在回合内开启一个引用桶；退出时恢复外层上下文。"""
    bucket: list[SearchHit] = []
    token = _CITATION_BUCKET.set(bucket)
    try:
        yield bucket
    finally:
        _CITATION_BUCKET.reset(token)


def _record_citations(hits: list[SearchHit]) -> None:
    bucket = _CITATION_BUCKET.get()
    if bucket is not None:
        bucket.extend(hits)


def citation_payload(hits: list[SearchHit], *, max_items: int = 8, snippet_chars: int = 240) -> list[dict]:
    """把检索命中整理成可直接下发给前端的引用结构（按片段去重）。"""
    payload: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        key = hit.chunk_id or f"{hit.source}|{hit.start_line}|{hit.end_line}"
        if key in seen:
            continue
        seen.add(key)
        text = " ".join(hit.text.split())
        if len(text) > snippet_chars:
            text = text[:snippet_chars].rstrip() + "…"
        payload.append(
            {
                "source": hit.source or "未知",
                "heading": hit.heading,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "score": round(float(hit.score), 4),
                "snippet": text,
                "chunk_id": hit.chunk_id,
            }
        )
        if len(payload) >= max_items:
            break
    return payload


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        raise ValueError("文件看起来是二进制内容，无法作为文本摄入")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _source_name(path: Path) -> str:
    try:
        return path.relative_to(workspace_root()).as_posix()
    except ValueError:
        return str(path)


def _iter_document_files(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES
    ]
    return sorted(files, key=lambda item: item.as_posix())


@register_tool
def knowledge_ingest(path: str) -> str:
    """把工作区内的文本/Markdown 文件加入知识库；之后可用 knowledge_search 检索。"""
    full = resolve_workspace_path(path, must_exist=True)
    if full.is_dir():
        return f"{path} 是目录，请使用 knowledge_ingest_dir 批量摄入。"
    text = _read_text_file(full)
    if len(text) > _MAX_INGEST_CHARS:
        raise ValueError(f"文档超过 {_MAX_INGEST_CHARS} 字符，请先拆分后再摄入")
    kb = get_knowledge_base()
    source = _source_name(full)
    count = asyncio.run(kb.ingest(text, source=source, replace_source=True))
    suffix = "（索引写入失败，当前仅内存可用）" if kb.last_persist_error else ""
    return (
        f"已从 {source} 摄入 {count} 个片段，"
        f"知识库当前共 {len(kb)} 个片段{suffix}。"
    )


@register_tool
def knowledge_ingest_dir(path: str = ".", recursive: bool = False) -> str:
    """批量摄入工作区内某个目录下的文本/代码文件（单个文件上限 200 万字符）。"""
    full = resolve_workspace_path(path, must_exist=True)
    if not full.is_dir():
        return f"{path} 不是目录；单个文件请使用 knowledge_ingest。"

    files = _iter_document_files(full, recursive)
    if not files:
        return (
            f"目录 {path} 下没有可摄入的文本文件（支持 {', '.join(sorted(_TEXT_SUFFIXES))}）。"
        )

    truncated = len(files) > _MAX_DIR_FILES
    files = files[:_MAX_DIR_FILES]
    kb = get_knowledge_base()

    async def _run() -> tuple[int, int, list[str]]:
        ingested = 0
        chunks = 0
        skipped: list[str] = []
        for file in files:
            source = _source_name(file)
            try:
                text = _read_text_file(file)
                if not text.strip():
                    skipped.append(f"{source}（空文件）")
                    continue
                if len(text) > _MAX_INGEST_CHARS:
                    skipped.append(f"{source}（超过 {_MAX_INGEST_CHARS} 字符）")
                    continue
                chunks += await kb.ingest(text, source=source, replace_source=True)
                ingested += 1
            except Exception as exc:  # 单个文件失败不应中断整批摄入
                skipped.append(f"{source}（{type(exc).__name__}: {exc}）")
        return ingested, chunks, skipped

    ingested, chunks, skipped = asyncio.run(_run())
    lines = [
        f"已从 {path} 摄入 {ingested}/{len(files)} 个文件，共 {chunks} 个片段；"
        f"知识库当前共 {len(kb)} 个片段。"
    ]
    if truncated:
        lines.append(f"文件数超过 {_MAX_DIR_FILES}，只处理了前 {_MAX_DIR_FILES} 个。")
    if kb.last_persist_error:
        lines.append("索引写入失败，当前仅内存可用。")
    if skipped:
        preview = "；".join(skipped[:5])
        more = f"（另有 {len(skipped) - 5} 个）" if len(skipped) > 5 else ""
        lines.append(f"跳过：{preview}{more}")
    return "\n".join(lines)


@register_tool
def knowledge_search(query: str, top_k: int = 0) -> str:
    """在知识库中混合检索（关键词 + 向量 + MMR）最相关的原文片段；top_k=0 用配置默认值。"""
    kb = get_knowledge_base()
    if len(kb) == 0:
        return "知识库为空，请先用 knowledge_ingest 或 knowledge_ingest_dir 摄入文档。"
    limit = kb.default_top_k if top_k <= 0 else top_k
    hits = asyncio.run(kb.search_hits(query, top_k=limit))
    if not hits:
        return "没有检索到相关片段。"
    _record_citations(hits)
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        location = hit.source or "未知"
        if hit.heading:
            location += f" / {hit.heading}"
        if hit.start_line and hit.end_line:
            if hit.start_line == hit.end_line:
                location += f" 第{hit.start_line}行"
            else:
                location += f" 第{hit.start_line}-{hit.end_line}行"
        blocks.append(
            f"[片段{rank} 来源:{location} 相关度:{hit.score:.2f}]\n{hit.text}"
        )
    note = "（嵌入服务不可用，已退化为关键词检索）" if kb.last_search_degraded else ""
    return "\n\n".join(blocks) + (f"\n\n{note}" if note else "")


@register_tool
def knowledge_list() -> str:
    """列出知识库中的全部来源、片段数与更新时间。"""
    kb = get_knowledge_base()
    sources = kb.list_sources()
    if not sources:
        return "知识库为空。"
    lines = [f"知识库共 {len(sources)} 个来源、{len(kb)} 个片段："]
    for item in sources:
        updated = str(item["updated_at"]) or "未知"
        lines.append(
            f"- {item['source']}：{item['chunks']} 片段，"
            f"{item['chars']} 字符，更新于 {updated}"
        )
    return "\n".join(lines)


@register_tool(requires_approval=True)
def knowledge_delete(source: str) -> str:
    """按来源删除知识库中的全部片段（来源名可用 knowledge_list 查看）。"""
    source = (source or "").strip()
    if not source:
        return "knowledge_delete 失败：source 不能为空。"
    if source in {"(未命名)", "未命名"}:
        source = ""
    kb = get_knowledge_base()
    removed = asyncio.run(kb.delete_source(source))
    if not removed:
        return f"没有找到来源 {source or '(未命名)'!r} 的片段。"
    suffix = "（索引写入失败，当前仅内存生效）" if kb.last_persist_error else ""
    return (
        f"已删除来源 {source or '(未命名)'!r} 的 {removed} 个片段，"
        f"知识库当前共 {len(kb)} 个片段{suffix}。"
    )


@register_tool
def knowledge_stats() -> str:
    """查看知识库索引概况（来源数、片段数、向量维度、持久化位置等）。"""
    kb = get_knowledge_base()
    stats = kb.stats()
    index_path = stats["index_path"] or "未启用持久化（仅内存）"
    lines = [
        f"来源: {stats['sources']}，片段: {stats['chunks']}，字符: {stats['chars']}",
        f"嵌入: {stats['embedder']}，维度: {stats['dims'] or '未知'}",
        f"切块: {stats['chunk_size']} 字 / 重叠 {stats['overlap']} 字",
        f"检索: 词法权重 {stats['lexical_weight']:.2f}，MMR λ {stats['mmr_lambda']:.2f}，"
        f"最低分 {stats['min_score']:.2f}",
        f"索引: {index_path}",
    ]
    if stats["needs_reindex"]:
        lines.append(f"有 {stats['missing_vectors']} 个片段缺向量，下次检索时自动补算。")
    if kb.last_persist_error:
        lines.append(f"最近一次索引写入失败: {kb.last_persist_error}")
    return "\n".join(lines)
