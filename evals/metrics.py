"""评测结果的匹配规则与五项指标聚合。

匹配全部是确定性的字符串/结构判断，不调用模型，因此同一份结果可以反复
复算、对比历史版本。指标按用例和调用两个粒度统计，避免长会话把比率带偏。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import Expectations


@dataclass
class CaseResult:
    case_id: str
    category: str
    tags: list[str] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    answer_ok: bool = False
    file_checks_ok: bool = True
    required_tools_ok: bool = True
    tools: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    tool_success: int = 0
    failed_tools: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    citation_ok: bool | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    citation_expected: int = 0
    citation_matched_returned: int = 0
    citation_matched_expected: int = 0
    prompt_tokens: list[int] = field(default_factory=list)
    prompt_budget: int = 0
    overflow: bool = False
    compacted: bool = False
    compacted_ok: bool = True
    todos: list[dict] = field(default_factory=list)
    todo_max_items: int = 0
    todo_ok: bool = True
    recovery_case: bool = False
    recovered: bool | None = None
    injected_failures: int = 0
    duration_ms: float = 0.0
    ok: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _fold(text: str) -> str:
    return text.casefold()


_REFUSAL_MARKERS = (
    "无法",
    "不能",
    "不会",
    "拒绝",
    "不便",
    "无权",
    "没有权限",
    "不具备",
    "不提供",
    "不掌握",
    "未找到",
    "没有找到",
    "没有相关",
    "未提及",
    "没有提及",
    "不存在",
    "不知道",
    "不清楚",
    "超出",
    "没有",
    "未提供",
    "无法确认",
    "未公开",
    "不公开",
    "不应",
    "反对",
    "不宜",
    "不成立",
)

_CLARIFY_HINTS = (
    "请提供",
    "请补充",
    "请确认",
    "请说明",
    "请告诉",
    "是指",
    "哪一个",
    "哪一份",
    "哪一种",
    "哪份",
    "哪种",
    "具体是",
    "能否提供",
    "能否说明",
    "需要您",
    "需要你",
    "告诉我",
    "给我",
    "发我",
    "请描述",
)


def match_answer(answer: str, expect: Expectations) -> bool:
    """按用例声明的关键词规则判断回答是否正确。

    - ``answer_exclude`` 命中任意一项即失败；
    - ``answer_all`` 中每一项都必须出现；
    - ``answer_any`` 是若干组，任一组内所有关键词都出现即通过；
    - ``answer_max_chars`` 用于约束"只输出 X"这类指令遵循用例；
    - ``refusal`` 要求回答表达拒绝/不知道，``clarify`` 要求追问缺失信息。
    """
    text = _fold(answer or "")
    if not text.strip():
        return False
    if expect.answer_exclude and any(
        _fold(item) in text for item in expect.answer_exclude
    ):
        return False
    if expect.answer_max_chars and len((answer or "").strip()) > expect.answer_max_chars:
        return False
    if expect.answer_all and not all(_fold(item) in text for item in expect.answer_all):
        return False
    if expect.answer_any and not any(
        all(_fold(item) in text for item in group) for group in expect.answer_any
    ):
        return False
    if expect.refusal and not any(
        _fold(item) in text for item in _REFUSAL_MARKERS
    ):
        return False
    if expect.clarify and not (
        "?" in (answer or "")
        or "？" in (answer or "")
        or any(_fold(item) in text for item in _CLARIFY_HINTS)
    ):
        return False
    return bool(
        expect.answer_all or expect.answer_any or expect.refusal or expect.clarify
    )


def _source_matches(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    return Path(actual).name == Path(expected).name


def citation_matches(expected, actual: dict) -> bool:
    """来源一致（允许路径差异），且可选内容片段出现在返回的 snippet/标题里。"""
    if not _source_matches(expected.source, str(actual.get("source") or "")):
        return False
    if not expected.contains:
        return True
    haystack = f"{actual.get('snippet') or ''}\n{actual.get('heading') or ''}"
    return _fold(expected.contains) in _fold(haystack)


def score_citations(
    citations: list[dict], expect: Expectations
) -> tuple[bool | None, float | None, float | None, int, int]:
    """返回 ``(是否全部命中, 精确率, 召回率, 命中返回条数, 命中期望条数)``。"""
    if not expect.citations:
        return None, None, None, 0, 0
    matched_returned = sum(
        1
        for citation in citations
        if any(citation_matches(item, citation) for item in expect.citations)
    )
    matched_expected = sum(
        1
        for item in expect.citations
        if any(citation_matches(item, citation) for citation in citations)
    )
    precision = matched_returned / len(citations) if citations else 0.0
    recall = matched_expected / len(expect.citations)
    return (
        matched_expected == len(expect.citations),
        round(precision, 4),
        round(recall, 4),
        matched_returned,
        matched_expected,
    )


def tool_events_summary(
    events: list[dict],
) -> tuple[int, int, list[str], dict[str, int], list[str]]:
    """从 tool_end 事件统计调用数、成功数、失败工具名与各工具次数。"""
    calls = 0
    success = 0
    failed: list[str] = []
    counts: dict[str, int] = {}
    denied: list[str] = []
    for event in events:
        if event.get("type") != "tool_end":
            continue
        calls += 1
        name = str(event.get("name") or "?")
        counts[name] = counts.get(name, 0) + 1
        if event.get("ok") and not event.get("denied"):
            success += 1
        else:
            failed.append(name)
            if event.get("denied"):
                denied.append(name)
    return calls, success, failed, counts, denied


def check_tool_expectations(
    expect: Expectations,
    counts: dict[str, int],
    failed: list[str],
) -> bool:
    if any(name not in counts for name in expect.tools):
        return False
    if any(counts.get(name, 0) < required for name, required in expect.min_tool_calls.items()):
        return False
    if any(name in counts for name in expect.no_tools):
        return False
    if any(name not in failed for name in expect.failed_tools):
        return False
    return True


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def aggregate(results: list[CaseResult], prompt_budget: int) -> dict:
    """把逐用例结果聚合成五项核心指标及辅助计数。

    ``prompt_budget`` 为本轮实际使用的 token 预算；用例内任何一次模型调用的
    估算 token 超过它，就记为一次上下文超限。
    """
    total = len(results)
    answer_ok = sum(1 for item in results if item.answer_ok)
    citation_cases = [item for item in results if item.citation_ok is not None]
    recovery_cases = [item for item in results if item.recovery_case]

    total_calls = sum(item.tool_calls for item in results)
    successful_calls = sum(item.tool_success for item in results)
    all_prompt_tokens = [tokens for item in results for tokens in item.prompt_tokens]
    overflow_calls = sum(
        1
        for item in results
        for tokens in item.prompt_tokens
        if tokens > (item.prompt_budget or prompt_budget)
    )
    overflow_cases = sum(1 for item in results if item.overflow)
    returned_citations = sum(len(item.citations) for item in results)
    matched_returned = sum(item.citation_matched_returned for item in results)
    matched_expected = sum(item.citation_matched_expected for item in results)
    expected_citations = sum(item.citation_expected for item in results)

    return {
        "total_cases": total,
        "answer_accuracy": _rate(answer_ok, total),
        "citation_accuracy": _rate(
            sum(1 for item in citation_cases if item.citation_ok), len(citation_cases)
        ),
        "tool_call_success_rate": _rate(successful_calls, total_calls),
        "context_overflow_rate": _rate(overflow_cases, total),
        "recovery_rate": _rate(
            sum(1 for item in recovery_cases if item.recovered), len(recovery_cases)
        ),
        "case_pass_rate": _rate(sum(1 for item in results if item.ok), total),
        # 辅助计数，便于判断比率背后的样本量。
        "answer_correct": answer_ok,
        "cases_passed": sum(1 for item in results if item.ok),
        "citation_cases": len(citation_cases),
        "citation_precision": _rate(matched_returned, returned_citations),
        "citation_recall": _rate(matched_expected, expected_citations),
        "tool_calls": total_calls,
        "tool_success": successful_calls,
        "prompt_calls": len(all_prompt_tokens),
        "overflow_calls": overflow_calls,
        "context_overflow_call_rate": _rate(overflow_calls, len(all_prompt_tokens)),
        "recovery_cases": len(recovery_cases),
        "recovered": sum(1 for item in recovery_cases if item.recovered),
        "prompt_budget": prompt_budget,
    }


def aggregate_by_category(results: list[CaseResult], prompt_budget: int) -> dict[str, dict]:
    """按 qa / rag / tools / context / recovery 分别聚合，定位退化来源。"""
    categories = sorted({item.category for item in results})
    return {
        category: aggregate(
            [item for item in results if item.category == category], prompt_budget
        )
        for category in categories
    }


def aggregate_by_tag(results: list[CaseResult], prompt_budget: int) -> dict[str, dict]:
    """按 tag 分别聚合，让拒答、幻觉、难度分层各自有独立通过率。

    一条用例可带多个 tag，会同时计入每个 tag 的统计，因此各 tag 的样本数
    之和可以大于总用例数。
    """
    tags = sorted({tag for item in results for tag in item.tags})
    return {
        tag: aggregate([item for item in results if tag in item.tags], prompt_budget)
        for tag in tags
    }