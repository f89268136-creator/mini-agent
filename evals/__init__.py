"""固定评测集：用例模型、五项指标、真实 MiniBot 执行器与报告。"""

from .metrics import (
    CaseResult,
    aggregate,
    aggregate_by_category,
    aggregate_by_tag,
    check_tool_expectations,
    match_answer,
    score_citations,
    tool_events_summary,
)
from .runner import EvalOptions, MeteredProvider, render_markdown, run_case, run_suite, write_report
from .schema import CATEGORIES, CitationExpectation, EvalCase, Expectations, Setup, load_cases

__all__ = [
    "CATEGORIES",
    "CaseResult",
    "CitationExpectation",
    "EvalCase",
    "EvalOptions",
    "Expectations",
    "MeteredProvider",
    "Setup",
    "aggregate",
    "aggregate_by_category",
    "aggregate_by_tag",
    "check_tool_expectations",
    "load_cases",
    "match_answer",
    "render_markdown",
    "run_case",
    "run_suite",
    "score_citations",
    "tool_events_summary",
    "write_report",
]