"""``python -m mini_agent.evals``：固定评测集的命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from ..config import ConfigError, load_config
from .runner import EvalOptions, run_suite, write_report
from .schema import EvalCase, load_cases


def _select(cases: list[EvalCase], only: str, limit: int) -> list[EvalCase]:
    selected = cases
    prefixes = [item.strip() for item in only.split(",") if item.strip()]
    if prefixes:
        selected = [
            case
            for case in selected
            if any(
                case.id.startswith(prefix)
                or case.category == prefix
                or prefix in case.tags
                for prefix in prefixes
            )
        ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _print_metrics(metrics: dict) -> None:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    print(
        f"回答正确率     {percent(metrics['answer_accuracy'])} "
        f"({metrics['answer_correct']}/{metrics['total_cases']})"
    )
    print(
        f"引用准确率     {percent(metrics['citation_accuracy'])} "
        f"({metrics['citation_cases']} 个用例；精确率 "
        f"{percent(metrics['citation_precision'])}，召回率 "
        f"{percent(metrics['citation_recall'])})"
    )
    print(
        f"工具调用成功率 {percent(metrics['tool_call_success_rate'])} "
        f"({metrics['tool_success']}/{metrics['tool_calls']})"
    )
    print(
        f"上下文超限率   {percent(metrics['context_overflow_rate'])} "
        f"({metrics['overflow_calls']}/{metrics['prompt_calls']} 次调用)"
    )
    print(
        f"失败恢复率     {percent(metrics['recovery_rate'])} "
        f"({metrics['recovered']}/{metrics['recovery_cases']})"
    )
    print(
        f"用例通过率     {percent(metrics['case_pass_rate'])} "
        f"({metrics['cases_passed']}/{metrics['total_cases']})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mini_agent.evals",
        description="运行 75 条固定评测集，输出回答/引用/工具/上下文/恢复五项指标。",
    )
    parser.add_argument("--cases", help="用例 JSONL 路径（默认 mini_agent/evals/cases.jsonl）")
    parser.add_argument("--config", help="评测使用的配置文件路径（默认按常规规则查找）")
    parser.add_argument("--model", help="覆盖配置文件里的模型名")
    parser.add_argument(
        "--only",
        default="",
        help="只跑匹配的用例：逗号分隔的 id 前缀、分类或标签，例如 rag,tool-,rec-01",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多跑前 N 条（0 表示不限）")
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="单条用例超时秒数（默认 180）"
    )
    parser.add_argument("--keep", action="store_true", help="保留每条用例的临时目录")
    parser.add_argument(
        "--output",
        default="eval_reports",
        help="报告输出目录（默认 eval_reports）",
    )
    parser.add_argument("--no-report", action="store_true", help="不写 JSON/Markdown 报告")
    parser.add_argument("--dry-run", action="store_true", help="只加载并校验用例集")
    parser.add_argument("--list-cases", action="store_true", help="列出用例后退出")
    parser.add_argument("--verbose", action="store_true", help="显示运行日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("mini_agent.sdk").setLevel(logging.ERROR)
    logging.getLogger("mini_agent.evals").setLevel(logging.INFO)

    try:
        cases = load_cases(args.cases)
    except ValueError as exc:
        print(f"评测集不可用: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        counts = Counter(case.category for case in cases)
        print(f"评测集共 {len(cases)} 条：" + "，".join(
            f"{name} {counts[name]}" for name in sorted(counts)
        ))
        return 0

    selected = _select(cases, args.only, max(0, args.limit))
    if not selected:
        print("筛选后没有用例可运行。", file=sys.stderr)
        return 2

    if args.list_cases:
        for case in selected:
            print(f"{case.id}\t{case.category}\t{case.question}")
        return 0

    try:
        base = load_config(args.config)
    except ConfigError as exc:
        print(f"配置不可用: {exc}", file=sys.stderr)
        return 2
    if args.model:
        base.model = args.model

    options = EvalOptions(timeout=max(1.0, args.timeout), keep=args.keep)
    try:
        report = asyncio.run(run_suite(selected, base, options))
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return 130

    print()
    print(f"模型 {report['model']}，用例 {report['total_cases']} 条")
    print("-" * 56)
    _print_metrics(report["metrics"])

    if not args.no_report:
        json_path, markdown_path = write_report(report, Path(args.output))
        print("-" * 56)
        print(f"JSON 报告: {json_path}")
        print(f"Markdown 报告: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())