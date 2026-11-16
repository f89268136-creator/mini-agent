"""固定评测集的用例模型、加载与校验。

用例存放在 ``cases.jsonl``：一行一个 JSON，字段保持扁平，便于人工审阅和在
PR 里逐行 diff。这里只做静态校验，不访问模型、不写磁盘。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("qa", "rag", "tools", "context", "recovery")
FAILURE_MODES = ("", "provider_transient")
_CASE_FILE = Path(__file__).with_name("cases.jsonl")


@dataclass(frozen=True)
class CitationExpectation:
    source: str
    contains: str = ""


@dataclass(frozen=True)
class Expectations:
    # answer_all 中每一项都必须出现；answer_any 是若干"必须同时出现"的组，命中任一组即可。
    answer_all: tuple[str, ...] = ()
    answer_any: tuple[tuple[str, ...], ...] = ()
    answer_exclude: tuple[str, ...] = ()
    answer_max_chars: int = 0
    # 拒答型期望：回答必须表达"无法/不能/未找到"等拒绝语义，用于安全与幻觉用例。
    refusal: bool = False
    # 澄清型期望：回答必须追问缺失信息，用于意图模糊、参数缺失用例。
    clarify: bool = False
    # 必须至少出现一次的工具名；min_tool_calls 进一步要求出现次数。
    tools: tuple[str, ...] = ()
    min_tool_calls: dict[str, int] = field(default_factory=dict)
    # 必须观察到失败（ok=false）的工具；用于失败恢复用例。
    failed_tools: tuple[str, ...] = ()
    no_tools: tuple[str, ...] = ()
    citations: tuple[CitationExpectation, ...] = ()
    # 用例结束后检查工作区文件：files 要求包含，files_exclude 要求不包含。
    files: dict[str, str] = field(default_factory=dict)
    files_exclude: dict[str, str] = field(default_factory=dict)
    compacted: bool = False
    # 任务清单跟踪：结束时必须存在清单、全部完成、没有丢项，项数不低于下限。
    todo_all_completed: bool = False
    todo_no_drop: bool = False
    todo_min_items: int = 0

    @property
    def has_answer_rule(self) -> bool:
        return bool(
            self.answer_all or self.answer_any or self.refusal or self.clarify
        )


@dataclass(frozen=True)
class HistoryFiller:
    turns: int = 0
    chars: int = 300


@dataclass(frozen=True)
class Setup:
    files: dict[str, str] = field(default_factory=dict)
    knowledge: tuple[dict[str, str], ...] = ()
    memory: tuple[str, ...] = ()
    # 先按 filler 生成多轮闲聊，再追加 history；用于稳定构造超长上下文。
    history_filler: HistoryFiller = field(default_factory=HistoryFiller)
    history: tuple[dict[str, str], ...] = ()
    prompt_max_tokens: int = 0
    failure: str = ""


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    setup: Setup = field(default_factory=Setup)
    expect: Expectations = field(default_factory=Expectations)
    tags: tuple[str, ...] = ()

    @property
    def recovery_case(self) -> bool:
        return bool(self.expect.failed_tools or self.setup.failure)


def _tuple_of_str(value: object, field_name: str, *, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{where}: {field_name} 必须是字符串数组")
    return tuple(value)


def _dict_of_str(value: object, field_name: str, *, where: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{where}: {field_name} 必须是字符串到字符串的对象")
    return dict(value)


def _safe_relative(path: str, *, where: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{where}: 文件路径必须是工作区内的相对路径: {path!r}")
    return path


def _parse_expectations(raw: object, *, where: str) -> Expectations:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expect 必须是对象")
    answer_any_raw = raw.get("answer_any") or []
    if not isinstance(answer_any_raw, list) or not all(
        isinstance(group, list) and all(isinstance(item, str) for item in group)
        for group in answer_any_raw
    ):
        raise ValueError(f"{where}: answer_any 必须是二维字符串数组")
    citations_raw = raw.get("citations") or []
    if not isinstance(citations_raw, list):
        raise ValueError(f"{where}: citations 必须是数组")
    citations: list[CitationExpectation] = []
    for item in citations_raw:
        if not isinstance(item, dict) or not isinstance(item.get("source"), str):
            raise ValueError(f"{where}: citation 必须包含字符串 source")
        citations.append(
            CitationExpectation(
                source=item["source"],
                contains=str(item.get("contains") or ""),
            )
        )
    min_calls_raw = raw.get("min_tool_calls") or {}
    if not isinstance(min_calls_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and value > 0
        for key, value in min_calls_raw.items()
    ):
        raise ValueError(f"{where}: min_tool_calls 必须是 工具名 -> 正整数")
    files = _dict_of_str(raw.get("files"), "files", where=where)
    for path in files:
        _safe_relative(path, where=where)
    files_exclude = _dict_of_str(raw.get("files_exclude"), "files_exclude", where=where)
    for path in files_exclude:
        _safe_relative(path, where=where)

    max_chars = raw.get("answer_max_chars") or 0
    if not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError(f"{where}: answer_max_chars 必须是非负整数")

    todo_min_items = raw.get("todo_min_items") or 0
    if not isinstance(todo_min_items, int) or todo_min_items < 0:
        raise ValueError(f"{where}: todo_min_items 必须是非负整数")

    return Expectations(
        answer_all=_tuple_of_str(raw.get("answer_all"), "answer_all", where=where),
        answer_any=tuple(tuple(group) for group in answer_any_raw),
        answer_exclude=_tuple_of_str(
            raw.get("answer_exclude"), "answer_exclude", where=where
        ),
        answer_max_chars=max_chars,
        refusal=bool(raw.get("refusal") or False),
        clarify=bool(raw.get("clarify") or False),
        tools=_tuple_of_str(raw.get("tools"), "tools", where=where),
        min_tool_calls=dict(min_calls_raw),
        failed_tools=_tuple_of_str(raw.get("failed_tools"), "failed_tools", where=where),
        no_tools=_tuple_of_str(raw.get("no_tools"), "no_tools", where=where),
        citations=tuple(citations),
        files=files,
        files_exclude=files_exclude,
        compacted=bool(raw.get("compacted") or False),
        todo_all_completed=bool(raw.get("todo_all_completed") or False),
        todo_no_drop=bool(raw.get("todo_no_drop") or False),
        todo_min_items=todo_min_items,
    )


def _parse_setup(raw: object, *, where: str) -> Setup:
    if raw is None:
        return Setup()
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: setup 必须是对象")
    files = _dict_of_str(raw.get("files"), "files", where=where)
    for path in files:
        _safe_relative(path, where=where)

    knowledge_raw = raw.get("knowledge") or []
    if not isinstance(knowledge_raw, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("source"), str)
        and isinstance(item.get("text"), str)
        for item in knowledge_raw
    ):
        raise ValueError(f"{where}: knowledge 必须是 {{source, text}} 数组")

    filler_raw = raw.get("history_filler") or {}
    if not isinstance(filler_raw, dict):
        raise ValueError(f"{where}: history_filler 必须是对象")
    filler = HistoryFiller(
        turns=int(filler_raw.get("turns") or 0),
        chars=int(filler_raw.get("chars") or 300),
    )
    if filler.turns < 0 or filler.chars < 0:
        raise ValueError(f"{where}: history_filler 不能为负")

    history_raw = raw.get("history") or []
    if not isinstance(history_raw, list) or not all(
        isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
        for item in history_raw
    ):
        raise ValueError(f"{where}: history 必须是 {{role, content}} 数组")

    prompt_max_tokens = raw.get("prompt_max_tokens") or 0
    if not isinstance(prompt_max_tokens, int) or prompt_max_tokens < 0:
        raise ValueError(f"{where}: prompt_max_tokens 必须是非负整数")

    failure = str(raw.get("failure") or "")
    if failure not in FAILURE_MODES:
        raise ValueError(f"{where}: failure 必须是 {FAILURE_MODES!r} 之一")

    return Setup(
        files=files,
        knowledge=tuple(dict(item) for item in knowledge_raw),
        memory=_tuple_of_str(raw.get("memory"), "memory", where=where),
        history_filler=filler,
        history=tuple(dict(item) for item in history_raw),
        prompt_max_tokens=prompt_max_tokens,
        failure=failure,
    )


def parse_case(raw: object, *, where: str = "case") -> EvalCase:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: 必须是对象")
    case_id = raw.get("id")
    category = raw.get("category")
    question = raw.get("question")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{where}: id 不能为空")
    if category not in CATEGORIES:
        raise ValueError(f"{where}: category 必须是 {CATEGORIES!r} 之一")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{where}: question 不能为空")

    expectations = _parse_expectations(raw.get("expect"), where=where)
    if not expectations.has_answer_rule:
        raise ValueError(f"{where}: 必须提供 answer_all 或 answer_any")
    if category == "rag" and not expectations.citations:
        raise ValueError(f"{where}: rag 用例必须提供 citations 期望")
    if category == "recovery" and not (
        expectations.failed_tools or str((raw.get("setup") or {}).get("failure") or "")
    ):
        raise ValueError(f"{where}: recovery 用例必须注入故障或期望工具失败")

    return EvalCase(
        id=case_id,
        category=category,
        question=question,
        setup=_parse_setup(raw.get("setup"), where=where),
        expect=expectations,
        tags=_tuple_of_str(raw.get("tags"), "tags", where=where),
    )


def load_cases(path: str | Path | None = None) -> list[EvalCase]:
    """读取并校验固定评测集；空行与 ``#`` 注释会被忽略。"""
    source = Path(path) if path is not None else _CASE_FILE
    cases: list[EvalCase] = []
    seen: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取评测集 {source}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        where = f"{source.name}:{line_number}"
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: JSON 解析失败: {exc}") from exc
        case = parse_case(raw, where=where)
        if case.id in seen:
            raise ValueError(f"{where}: 用例 id 重复: {case.id}")
        seen.add(case.id)
        cases.append(case)

    if not 50 <= len(cases) <= 100:
        raise ValueError(f"评测集必须包含 50~100 条用例，当前 {len(cases)} 条")
    return cases