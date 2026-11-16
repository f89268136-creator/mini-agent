"""评测执行器：为每条用例准备隔离环境，调用真实的 MiniBot，再按规则打分。

隔离策略：一条用例一个临时工作区与一套会话/记忆/清单/索引目录，跑完即删。
这样用例之间不会共享文件、历史或检索片段，失败也能单独复现（``--keep``）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from ..agent.loop import _SUMMARY_SYSTEM_PROMPT
from ..agent.token_budget import estimate_messages_tokens
from ..agent.tools.knowledge import get_knowledge_base
from ..config import MiniConfig
from ..providers.base import DeltaCallback, LLMProvider, LLMResponse, Message
from ..sdk import MiniBot
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
from .schema import EvalCase

logger = logging.getLogger(__name__)

_FILLER_BASE = "这是评测历史填充内容，用来稳定构造较长的上下文，不包含任何待回答的事实。"


@dataclass
class EvalOptions:
    timeout: float = 180.0
    keep: bool = False


class MeteredProvider(LLMProvider):
    """记录每次发给模型的 prompt 估算 token；压缩摘要调用单独排除。"""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.prompt_tokens: list[int] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        if not _is_summary_call(messages):
            self.prompt_tokens.append(estimate_messages_tokens(messages))
        return await self.inner.chat(messages, tools, on_delta=on_delta)

    async def aclose(self) -> None:
        await self.inner.aclose()


class _InjectedInner(LLMProvider):
    """在第一次模型调用前注入一次瞬时超时，用于验证真实重试链路。"""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.failures = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        if self.failures == 0:
            self.failures += 1
            raise TimeoutError("评测注入的瞬时超时")
        return await self.inner.chat(messages, tools, on_delta=on_delta)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _is_summary_call(messages: list[Message]) -> bool:
    return bool(
        messages
        and messages[0].role == "system"
        and (messages[0].content or "") == _SUMMARY_SYSTEM_PROMPT
    )


def _session_key(case: EvalCase) -> str:
    return f"eval:{case.id}"


def _filler_text(chars: int) -> str:
    if chars <= 0:
        return ""
    repeats = chars // len(_FILLER_BASE) + 1
    return (_FILLER_BASE * repeats)[:chars]


def _build_config(base: MiniConfig, case: EvalCase, root: Path) -> MiniConfig:
    workspace = root / "workspace"
    budget = case.setup.prompt_max_tokens or base.prompt_max_tokens
    config = replace(
        base,
        workspace=str(workspace),
        sessions_dir=str(root / "sessions"),
        memory_dir=str(root / "memory"),
        mem0_dir=str(root / "mem0"),
        todos_dir=str(root / "todos"),
        knowledge_dir=str(root / "knowledge"),
        memory_backend="local",
        memory_enabled=True,
        todos_enabled=True,
        skills_enabled=False,
        embedding_backend="hashing",
        embedding_model=None,
        channels=[],
        stream=False,
        tool_approval="off",
        allow_shell=False,
        allow_private_urls=False,
        multi_agent_enabled=False,
        langfuse_enabled=False,
        retry_base_delay=0.0,
        history_limit=max(200, base.history_limit),
        prompt_max_tokens=budget,
        context_compact_ratio=0.8,
        context_keep_turns=4,
        context_summary_max_chars=min(base.context_summary_max_chars, 1000),
        max_iterations=max(base.max_iterations, 8),
    )
    config.validate()
    return config


def _write_setup_files(root: Path, case: EvalCase) -> None:
    for relative, content in case.setup.files.items():
        target = root / "workspace" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


async def _seed_session(bot: MiniBot, case: EvalCase, key: str) -> None:
    session = bot.sessions.get_or_create(key)
    filler = case.setup.history_filler
    text = _filler_text(filler.chars)
    for index in range(filler.turns):
        session.add_message(
            Message(role="user", content=f"第 {index + 1} 轮闲聊：{text}")
        )
        session.add_message(
            Message(role="assistant", content=f"收到第 {index + 1} 轮内容。{text}")
        )
    for item in case.setup.history:
        session.add_message(Message(role=item["role"], content=item["content"]))
    if session.messages:
        await bot.sessions.save(session)


async def _seed_memory(bot: MiniBot, case: EvalCase, key: str) -> None:
    for text in case.setup.memory:
        bot.memory.add(key, text)


async def _seed_knowledge(case: EvalCase) -> None:
    kb = get_knowledge_base()
    await kb.clear()
    for document in case.setup.knowledge:
        await kb.ingest(
            document["text"], source=document["source"], replace_source=True
        )


def _check_files(workspace: Path, case: EvalCase) -> bool:
    for relative, needle in case.expect.files.items():
        path = workspace / relative
        if not path.is_file():
            return False
        if needle not in path.read_text(encoding="utf-8", errors="replace"):
            return False
    for relative, needle in case.expect.files_exclude.items():
        path = workspace / relative
        if not path.is_file():
            continue
        if needle in path.read_text(encoding="utf-8", errors="replace"):
            return False
    return True


def _check_todos(
    case: EvalCase, todos: list[dict], max_items: int
) -> bool:
    expect = case.expect
    if not (expect.todo_all_completed or expect.todo_no_drop or expect.todo_min_items):
        return True
    if not todos:
        return False
    if expect.todo_all_completed and any(
        item.get("status") != "completed" for item in todos
    ):
        return False
    if expect.todo_no_drop and len(todos) < max_items:
        return False
    if len(todos) < expect.todo_min_items:
        return False
    return True


def _snapshot_todos(bot: MiniBot, key: str) -> list[dict]:
    store = getattr(bot, "todo", None)
    if store is None or not store.enabled:
        return []
    return [item.to_dict() for item in store.items(key)]


def _finish(
    result: CaseResult,
    case: EvalCase,
    *,
    events: list[dict],
    workspace: Path,
    compacted: bool,
    todos: list[dict],
    todo_max_items: int,
) -> None:
    result.answer_ok = match_answer(result.answer, case.expect)
    result.file_checks_ok = _check_files(workspace, case)
    result.todos = todos
    result.todo_max_items = todo_max_items
    result.todo_ok = _check_todos(case, todos, todo_max_items)

    calls, success, failed, counts, _denied = tool_events_summary(events)
    result.tools = [
        {
            key: event.get(key)
            for key in ("id", "name", "ok", "denied", "side_effect", "duration_ms")
        }
        for event in events
        if event.get("type") == "tool_end"
    ]
    result.tool_calls = calls
    result.tool_success = success
    result.failed_tools = failed
    result.required_tools_ok = check_tool_expectations(case.expect, counts, failed)

    (
        result.citation_ok,
        result.citation_precision,
        result.citation_recall,
        result.citation_matched_returned,
        result.citation_matched_expected,
    ) = score_citations(result.citations, case.expect)
    result.citation_expected = len(case.expect.citations)

    result.compacted = compacted
    result.compacted_ok = (not case.expect.compacted) or compacted
    result.overflow = bool(
        result.prompt_budget
        and any(tokens > result.prompt_budget for tokens in result.prompt_tokens)
    )

    if case.recovery_case:
        injected_ok = (
            case.setup.failure != "provider_transient"
            or result.injected_failures > 0
        )
        result.recovered = bool(
            not result.error
            and result.answer_ok
            and result.file_checks_ok
            and result.required_tools_ok
            and injected_ok
        )

    result.ok = bool(
        not result.error
        and result.answer_ok
        and result.file_checks_ok
        and result.required_tools_ok
        and result.todo_ok
        and result.citation_ok is not False
        and result.compacted_ok
    )


async def run_case(
    case: EvalCase,
    base: MiniConfig,
    options: EvalOptions,
) -> CaseResult:
    """在隔离环境里跑一条用例；任何异常都记录到结果而不是中断整套评测。"""
    started = time.perf_counter()
    root = Path(tempfile.mkdtemp(prefix=f"mini-agent-eval-{case.id}-"))
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    key = _session_key(case)
    budget = case.setup.prompt_max_tokens or base.prompt_max_tokens
    result = CaseResult(
        case_id=case.id,
        category=case.category,
        tags=list(case.tags),
        prompt_budget=budget,
        recovery_case=case.recovery_case,
    )
    events: list[dict] = []
    bot: MiniBot | None = None
    session = None
    todo_max_items = 0
    try:
        _write_setup_files(root, case)
        bot = MiniBot(_build_config(base, case, root))
        await _seed_session(bot, case, key)
        session = bot.sessions.get_or_create(key)
        await _seed_memory(bot, case, key)
        await _seed_knowledge(case)

        injected = None
        if case.setup.failure == "provider_transient":
            retrying = bot.runner.provider
            inner = getattr(retrying, "inner", None)
            if inner is not None:
                injected = _InjectedInner(inner)
                retrying.inner = injected
            else:  # pragma: no cover - build_provider 永远返回 RetryingProvider
                raise RuntimeError("provider 缺少 inner，无法注入瞬时故障")
        meter = MeteredProvider(bot.runner.provider)
        bot.runner.provider = meter

        async def on_event(event: dict) -> None:
            nonlocal todo_max_items
            events.append(event)
            if (
                event.get("type") == "tool_end"
                and event.get("name") == "todo_write"
                and event.get("ok")
                and not event.get("denied")
            ):
                todo_max_items = max(
                    todo_max_items, len(_snapshot_todos(bot, key))
                )

        outbound = await asyncio.wait_for(
            bot.loop.handle_text(
                channel="sdk",
                sender="eval",
                text=case.question,
                session_key=key,
                on_event=on_event,
            ),
            timeout=options.timeout,
        )
        result.answer = outbound.text or ""
        result.citations = list(outbound.citations or [])
        result.prompt_tokens = meter.prompt_tokens
        result.injected_failures = injected.failures if injected is not None else 0
    except Exception as exc:  # noqa: BLE001 - 单条用例失败不影响其余用例
        result.error = f"{type(exc).__name__}: {exc}"
        if bot is not None:
            result.prompt_tokens = getattr(bot.runner.provider, "prompt_tokens", [])
    finally:
        if bot is not None:
            todos = _snapshot_todos(bot, key)
            try:
                await bot.stop(graceful_timeout=0)
            except Exception:  # noqa: BLE001 - 关闭失败不应覆盖用例结果
                logger.exception("关闭评测实例失败: %s", case.id)
        else:
            todos = []
        result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        _finish(
            result,
            case,
            events=events,
            workspace=workspace,
            compacted=bool(session.summary) if session is not None else False,
            todos=todos,
            todo_max_items=todo_max_items,
        )
        if options.keep:
            logger.warning("保留用例目录: %s", root)
        else:
            shutil.rmtree(root, ignore_errors=True)
    return result


async def run_suite(
    cases: list[EvalCase],
    base: MiniConfig,
    options: EvalOptions | None = None,
) -> dict:
    """顺序执行全部用例（进程级工具策略/知识库只允许一套配置）。"""
    options = options or EvalOptions()
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        logger.warning("[%d/%d] %s", index, len(cases), case.id)
        results.append(await run_case(case, base, options))
    budget = base.prompt_max_tokens
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": base.model,
        "api_base": base.api_base,
        "total_cases": len(cases),
        "metrics": aggregate(results, budget),
        "by_category": aggregate_by_category(results, budget),
        "by_tag": aggregate_by_tag(results, budget),
        "cases": [item.as_dict() for item in results],
    }


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _metric(count: int | float, total: int | float) -> str:
    return f"{int(count)}/{int(total)}"


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Mini Agent 固定评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 模型：{report['model']}",
        f"- 端点：{report['api_base']}",
        f"- 用例数：{report['total_cases']}",
        "",
        "## 五项核心指标",
        "",
        "| 指标 | 数值 | 计数 |",
        "| --- | --- | --- |",
        f"| 回答正确率 | {_percent(metrics['answer_accuracy'])} | "
        f"{_metric(metrics['answer_correct'], metrics['total_cases'])} |",
        f"| 引用准确率 | {_percent(metrics['citation_accuracy'])} | "
        f"{_metric(sum(1 for c in report['cases'] if c['citation_ok']), metrics['citation_cases'])} |",
        f"| 工具调用成功率 | {_percent(metrics['tool_call_success_rate'])} | "
        f"{_metric(metrics['tool_success'], metrics['tool_calls'])} |",
        f"| 上下文超限率 | {_percent(metrics['context_overflow_rate'])} | "
        f"{_metric(sum(1 for c in report['cases'] if c['overflow']), metrics['total_cases'])} |",
        f"| 失败后可恢复率 | {_percent(metrics['recovery_rate'])} | "
        f"{_metric(metrics['recovered'], metrics['recovery_cases'])} |",
        f"| 用例通过率 | {_percent(metrics['case_pass_rate'])} | "
        f"{_metric(metrics['cases_passed'], metrics['total_cases'])} |",
        "",
        "## 分类明细",
        "",
        "| 分类 | 通过率 | 回答正确率 | 引用准确率 | 工具成功率 | 超限率 | 恢复率 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category, item in report["by_category"].items():
        lines.append(
            f"| {category} | {_percent(item['case_pass_rate'])} | "
            f"{_percent(item['answer_accuracy'])} | {_percent(item['citation_accuracy'])} | "
            f"{_percent(item['tool_call_success_rate'])} | "
            f"{_percent(item['context_overflow_rate'])} | {_percent(item['recovery_rate'])} |"
        )

    by_tag = report.get("by_tag") or {}
    if by_tag:
        lines.extend(
            [
                "",
                "## 分层与专题明细（按 tag）",
                "",
                "| tag | 用例数 | 通过率 | 回答正确率 | 引用准确率 | 工具成功率 | 超限率 | 恢复率 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for tag, item in by_tag.items():
            lines.append(
                f"| {tag} | {item['total_cases']} | {_percent(item['case_pass_rate'])} | "
                f"{_percent(item['answer_accuracy'])} | {_percent(item['citation_accuracy'])} | "
                f"{_percent(item['tool_call_success_rate'])} | "
                f"{_percent(item['context_overflow_rate'])} | {_percent(item['recovery_rate'])} |"
            )

    failed = [case for case in report["cases"] if not case["ok"]]
    lines.extend(["", f"## 未通过用例（{len(failed)}）", ""])
    if not failed:
        lines.append("全部用例通过。")
    else:
        lines.append("| 用例 | 类别 | 原因 | 回答摘录 |")
        lines.append("| --- | --- | --- | --- |")
        for case in failed[:30]:
            reasons = []
            if case["error"]:
                reasons.append(case["error"])
            if not case["answer_ok"]:
                reasons.append("回答不符")
            if not case["file_checks_ok"]:
                reasons.append("文件不符")
            if not case["required_tools_ok"]:
                reasons.append("工具调用不符")
            if not case["todo_ok"]:
                reasons.append("清单跟踪不符")
            if case["citation_ok"] is False:
                reasons.append("引用不符")
            if not case["compacted_ok"]:
                reasons.append("未触发压缩")
            excerpt = " ".join((case["answer"] or "").split())[:80]
            lines.append(
                f"| {case['case_id']} | {case['category']} | "
                f"{'；'.join(reasons) or '未知'} | {excerpt} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"eval-{stamp}.json"
    markdown_path = output_dir / f"eval-{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "EvalOptions",
    "MeteredProvider",
    "render_markdown",
    "run_case",
    "run_suite",
    "write_report",
]