"""固定评测集离线测试：用例校验、匹配规则、五项指标与执行器接线。

全部离线运行：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python -m unittest \
        mini_agent.tests.test_evals -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from mini_agent.config import MiniConfig
from mini_agent.agent.tools import configure_knowledge_base
from mini_agent.agent.tools.knowledge import citation_payload, get_knowledge_base
from mini_agent.providers.embeddings import HashingEmbedder
from mini_agent.evals import (
    CaseResult,
    EvalOptions,
    aggregate,
    aggregate_by_category,
    aggregate_by_tag,
    check_tool_expectations,
    load_cases,
    match_answer,
    score_citations,
    tool_events_summary,
)
from mini_agent.evals.__main__ import _select
from mini_agent.evals.runner import (
    MeteredProvider,
    _InjectedInner,
    _build_config,
    _check_files,
    _check_todos,
    render_markdown,
    run_case,
    write_report,
)
from mini_agent.evals.schema import CitationExpectation, Expectations, parse_case
from mini_agent.providers.base import LLMProvider, LLMResponse, Message
from mini_agent.providers.openai_compat import RetryingProvider


def _raw_case(**overrides) -> dict:
    case = {
        "id": "qa-test",
        "category": "qa",
        "question": "测试问题",
        "expect": {"answer_all": ["ok"]},
    }
    case.update(overrides)
    return case


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases()

    def test_size_and_category_coverage(self) -> None:
        self.assertGreaterEqual(len(self.cases), 50)
        self.assertLessEqual(len(self.cases), 100)
        counts = Counter(case.category for case in self.cases)
        self.assertEqual(
            counts,
            {"qa": 16, "rag": 14, "tools": 17, "context": 12, "recovery": 6},
        )

    def test_ids_are_unique(self) -> None:
        ids = [case.id for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_rag_cases_expect_citations(self) -> None:
        for case in self.cases:
            if case.category == "rag":
                self.assertTrue(case.expect.citations, case.id)

    def test_context_cases_expect_compaction(self) -> None:
        for case in self.cases:
            if case.category == "context":
                self.assertTrue(case.expect.compacted)
                self.assertGreater(case.setup.prompt_max_tokens, 0)

    def test_recovery_cases_inject_failure(self) -> None:
        for case in self.cases:
            if case.category == "recovery":
                self.assertTrue(case.setup.failure or case.expect.failed_tools, case.id)


class SchemaValidationTests(unittest.TestCase):
    def test_rejects_missing_answer_rule(self) -> None:
        with self.assertRaises(ValueError):
            parse_case(_raw_case(expect={"tools": ["read_file"]}))

    def test_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            parse_case(
                _raw_case(
                    setup={"files": {"../escape.txt": "x"}},
                )
            )

    def test_rejects_rag_without_citations(self) -> None:
        with self.assertRaises(ValueError):
            parse_case(_raw_case(category="rag", expect={"answer_all": ["x"]}))

    def test_rejects_recovery_without_failure(self) -> None:
        with self.assertRaises(ValueError):
            parse_case(_raw_case(category="recovery"))

    def test_todo_expectations_parse_and_validate(self) -> None:
        case = parse_case(
            _raw_case(
                expect={
                    "answer_all": ["ok"],
                    "todo_all_completed": True,
                    "todo_no_drop": True,
                    "todo_min_items": 2,
                }
            )
        )
        self.assertTrue(case.expect.todo_all_completed)
        self.assertTrue(case.expect.todo_no_drop)
        self.assertEqual(case.expect.todo_min_items, 2)
        with self.assertRaises(ValueError):
            parse_case(
                _raw_case(
                    expect={"answer_all": ["ok"], "todo_min_items": -1},
                )
            )

    def test_load_cases_rejects_too_few(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.jsonl"
            path.write_text(
                json.dumps(_raw_case(), ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_cases(path)


class AnswerMatchingTests(unittest.TestCase):
    def test_all_keywords_are_required(self) -> None:
        expect = Expectations(answer_all=("alpha", "beta"))
        self.assertTrue(match_answer("ALPHA and beta", expect))
        self.assertFalse(match_answer("alpha only", expect))

    def test_any_group_can_match(self) -> None:
        expect = Expectations(answer_any=(("推理", "行动"), ("reasoning", "action")))
        self.assertTrue(match_answer("靠 Reasoning 驱动 Action", expect))
        self.assertFalse(match_answer("只有推理", expect))

    def test_exclude_wins(self) -> None:
        expect = Expectations(answer_all=("是",), answer_exclude=("不是质数",))
        self.assertFalse(match_answer("97 不是质数", expect))
        self.assertTrue(match_answer("97 是质数", expect))

    def test_max_chars(self) -> None:
        expect = Expectations(answer_all=("ok",), answer_max_chars=4)
        self.assertTrue(match_answer("  OK ", expect))
        self.assertFalse(match_answer("OK 收到", expect))

    def test_empty_answer_fails(self) -> None:
        self.assertFalse(match_answer("", Expectations(answer_all=("ok",))))

    def test_refusal_requires_refusal_wording(self) -> None:
        expect = Expectations(refusal=True)
        self.assertTrue(match_answer("抱歉，我无法提供这类信息。", expect))
        self.assertTrue(match_answer("知识库里没有找到相关内容。", expect))
        self.assertFalse(match_answer("CEO 的生日是 5 月 1 日。", expect))

    def test_clarify_requires_follow_up_question(self) -> None:
        expect = Expectations(clarify=True)
        self.assertTrue(match_answer("请问您指的是哪个订单？", expect))
        self.assertTrue(match_answer("请提供具体的时间范围。", expect))
        self.assertTrue(match_answer("一句话把需求发我，我立刻开工。", expect))
        self.assertTrue(match_answer("你把数据给我后就能直接填出成品。", expect))
        self.assertFalse(match_answer("已经帮您处理好了。", expect))


class CitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expect = Expectations(
            answer_all=("x",),
            citations=(
                CitationExpectation(source="docs/deploy.md", contains="10%"),
                CitationExpectation(source="notes/ops.md"),
            ),
        )

    def test_source_path_and_snippet_matching(self) -> None:
        citations = [
            {
                "source": "/tmp/root/docs/deploy.md",
                "heading": "灰度",
                "snippet": "先放量 10% 的流量",
            },
            {"source": "notes/ops.md", "snippet": "备份时间"},
            {"source": "other.md", "snippet": "无关"},
        ]
        ok, precision, recall, matched_returned, matched_expected = score_citations(
            citations, self.expect
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(precision, 2 / 3, places=4)
        self.assertEqual(recall, 1.0)
        self.assertEqual(matched_returned, 2)
        self.assertEqual(matched_expected, 2)

    def test_missing_citation_reduces_recall(self) -> None:
        citations = [{"source": "docs/deploy.md", "snippet": "没有百分比"}]
        ok, precision, recall, _, matched_expected = score_citations(
            citations, self.expect
        )
        self.assertFalse(ok)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(matched_expected, 0)

    def test_no_expectation_returns_none(self) -> None:
        self.assertEqual(score_citations([], Expectations()), (None, None, None, 0, 0))


class ToolMetricTests(unittest.TestCase):
    def test_summary_counts_success_and_denial(self) -> None:
        events = [
            {"type": "tool_start", "name": "read_file"},
            {"type": "tool_end", "name": "read_file", "ok": True, "denied": False},
            {"type": "tool_end", "name": "write_file", "ok": False, "denied": True},
            {"type": "tool_end", "name": "grep", "ok": True, "denied": False},
        ]
        calls, success, failed, counts, denied = tool_events_summary(events)
        self.assertEqual((calls, success), (3, 2))
        self.assertEqual(failed, ["write_file"])
        self.assertEqual(denied, ["write_file"])
        self.assertEqual(counts, {"read_file": 1, "write_file": 1, "grep": 1})

    def test_expectations(self) -> None:
        expect = Expectations(
            tools=("read_file",),
            min_tool_calls={"read_file": 2},
            no_tools=("run_shell",),
            failed_tools=("write_file",),
        )
        self.assertTrue(
            check_tool_expectations(
                expect,
                {"read_file": 2, "write_file": 1},
                ["write_file"],
            )
        )
        self.assertFalse(
            check_tool_expectations(expect, {"read_file": 1}, [])
        )
        self.assertFalse(
            check_tool_expectations(
                expect, {"read_file": 2, "run_shell": 1}, ["write_file"]
            )
        )


class AggregateTests(unittest.TestCase):
    @staticmethod
    def _result(**overrides) -> CaseResult:
        values = {
            "case_id": "c",
            "category": "qa",
            "answer_ok": True,
            "tool_calls": 1,
            "tool_success": 1,
            "ok": True,
        }
        values.update(overrides)
        return CaseResult(**values)

    def test_five_metrics(self) -> None:
        results = [
            self._result(
                case_id="a",
                tool_calls=2,
                tool_success=2,
                citation_ok=True,
                citation_expected=1,
                citation_matched_returned=1,
                citation_matched_expected=1,
                citations=[{"source": "x"}],
                recovery_case=True,
                recovered=True,
            ),
            self._result(
                case_id="b",
                answer_ok=False,
                tool_calls=2,
                tool_success=0,
                overflow=True,
                recovery_case=True,
                recovered=False,
                ok=False,
            ),
        ]
        metrics = aggregate(results, prompt_budget=8000)
        self.assertEqual(metrics["total_cases"], 2)
        self.assertEqual(metrics["answer_accuracy"], 0.5)
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertEqual(metrics["tool_call_success_rate"], 0.5)
        self.assertEqual(metrics["context_overflow_rate"], 0.5)
        self.assertEqual(metrics["recovery_rate"], 0.5)
        self.assertEqual(metrics["cases_passed"], 1)

    def test_overflow_uses_per_case_budget(self) -> None:
        results = [
            self._result(case_id="small", prompt_budget=100, prompt_tokens=[90]),
            self._result(case_id="large", prompt_budget=1000, prompt_tokens=[500]),
        ]
        metrics = aggregate(results, prompt_budget=100)
        self.assertEqual(metrics["overflow_calls"], 0)
        results.append(
            self._result(case_id="over", prompt_budget=100, prompt_tokens=[101])
        )
        self.assertEqual(aggregate(results, 100)["overflow_calls"], 1)

    def test_by_category(self) -> None:
        results = [
            self._result(case_id="a", category="qa"),
            self._result(case_id="b", category="rag", answer_ok=False, ok=False),
        ]
        grouped = aggregate_by_category(results, 8000)
        self.assertEqual(grouped["qa"]["answer_accuracy"], 1.0)
        self.assertEqual(grouped["rag"]["answer_accuracy"], 0.0)

    def test_by_tag_counts_multi_tag_cases_in_each_group(self) -> None:
        results = [
            self._result(case_id="a", tags=["positive", "refusal"]),
            self._result(case_id="b", tags=["adversarial", "refusal"], answer_ok=False, ok=False),
        ]
        grouped = aggregate_by_tag(results, 8000)
        self.assertEqual(grouped["positive"]["case_pass_rate"], 1.0)
        self.assertEqual(grouped["adversarial"]["case_pass_rate"], 0.0)
        self.assertEqual(grouped["refusal"]["total_cases"], 2)
        self.assertEqual(grouped["refusal"]["case_pass_rate"], 0.5)


class TodoTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = parse_case(
            _raw_case(
                expect={
                    "answer_all": ["ok"],
                    "todo_all_completed": True,
                    "todo_no_drop": True,
                    "todo_min_items": 2,
                }
            )
        )

    @staticmethod
    def _items(*statuses: str) -> list[dict]:
        return [
            {"content": f"步骤 {index}", "status": status}
            for index, status in enumerate(statuses, start=1)
        ]

    def test_completed_todo_list_passes(self) -> None:
        todos = self._items("completed", "completed")
        self.assertTrue(_check_todos(self.case, todos, len(todos)))

    def test_pending_item_is_not_forgotten(self) -> None:
        todos = self._items("completed", "in_progress")
        self.assertFalse(_check_todos(self.case, todos, len(todos)))

    def test_dropped_step_is_detected(self) -> None:
        todos = self._items("completed", "completed")
        self.assertFalse(_check_todos(self.case, todos, 3))

    def test_missing_todo_list_fails(self) -> None:
        self.assertFalse(_check_todos(self.case, [], 0))

    def test_no_todo_expectation_ignores_state(self) -> None:
        self.assertTrue(_check_todos(parse_case(_raw_case()), [], 0))


class ProviderWrapperTests(unittest.TestCase):
    def test_metered_provider_skips_summary_calls(self) -> None:
        from mini_agent.agent.loop import _SUMMARY_SYSTEM_PROMPT

        class Inner(LLMProvider):
            async def chat(self, messages, tools=None, on_delta=None):
                return LLMResponse(content="ok")

        provider = MeteredProvider(Inner())
        asyncio.run(
            provider.chat([Message(role="system", content=_SUMMARY_SYSTEM_PROMPT)])
        )
        asyncio.run(provider.chat([Message(role="user", content="hello")]))
        self.assertEqual(len(provider.prompt_tokens), 1)
        self.assertGreater(provider.prompt_tokens[0], 0)

    def test_injected_failure_happens_once(self) -> None:
        class Inner(LLMProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, messages, tools=None, on_delta=None):
                self.calls += 1
                return LLMResponse(content="ok")

        inner = Inner()
        wrapper = _InjectedInner(inner)
        with self.assertRaises(TimeoutError):
            asyncio.run(wrapper.chat([Message(role="user", content="x")]))
        response = asyncio.run(wrapper.chat([Message(role="user", content="x")]))
        self.assertEqual(response.content, "ok")
        self.assertEqual(wrapper.failures, 1)
        self.assertEqual(inner.calls, 1)


class ConfigAndFileTests(unittest.TestCase):
    def test_build_config_isolates_paths_and_flags(self) -> None:
        case = parse_case(
            _raw_case(
                setup={"prompt_max_tokens": 4000},
                expect={"answer_all": ["ok"], "compacted": True},
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = _build_config(MiniConfig(api_key="k"), case, root)
            self.assertEqual(Path(config.workspace), root / "workspace")
            self.assertEqual(config.channels, [])
            self.assertEqual(config.tool_approval, "off")
            self.assertEqual(config.embedding_backend, "hashing")
            self.assertEqual(config.prompt_max_tokens, 4000)
            self.assertFalse(config.multi_agent_enabled)
            self.assertFalse(config.langfuse_enabled)

    def test_check_files(self) -> None:
        case = parse_case(
            _raw_case(
                expect={
                    "answer_all": ["ok"],
                    "files": {"out.txt": "HELLO"},
                    "files_exclude": {"out.txt": "draft"},
                }
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / "out.txt").write_text("HELLO", encoding="utf-8")
            self.assertTrue(_check_files(workspace, case))
            (workspace / "out.txt").write_text("HELLO draft", encoding="utf-8")
            self.assertFalse(_check_files(workspace, case))


class _StaticProvider(LLMProvider):
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def chat(self, messages, tools=None, on_delta=None):
        return LLMResponse(content=self.answer)


class _SummaryAwareProvider(LLMProvider):
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.summary_calls = 0

    async def chat(self, messages, tools=None, on_delta=None):
        from mini_agent.agent.loop import _SUMMARY_SYSTEM_PROMPT

        if messages and (messages[0].content or "") == _SUMMARY_SYSTEM_PROMPT:
            self.summary_calls += 1
            return LLMResponse(content="摘要：用户要求记住一个编号。")
        return LLMResponse(content=self.answer)


class RunCaseTests(unittest.TestCase):
    def test_run_case_end_to_end_with_fake_provider(self) -> None:
        case = parse_case(
            _raw_case(
                id="qa-e2e",
                question="请回答 TEST-OK",
                expect={"answer_all": ["test-ok"]},
            )
        )
        config = MiniConfig(
            api_key="offline-test",
            api_base="https://example.invalid/v1",
        )
        with mock.patch(
            "mini_agent.sdk.build_provider",
            lambda _config: _StaticProvider("TEST-OK"),
        ):
            result = asyncio.run(run_case(case, config, EvalOptions(timeout=30)))
        self.assertEqual(result.error, "")
        self.assertTrue(result.answer_ok)
        self.assertTrue(result.ok)
        self.assertEqual(result.tool_calls, 0)
        self.assertTrue(result.prompt_tokens)

    def test_context_compaction_is_detected_and_summary_call_not_metered(self) -> None:
        case = parse_case(
            _raw_case(
                id="ctx-e2e",
                category="context",
                question="刚才的编号是什么？",
                setup={
                    "history_filler": {"turns": 18, "chars": 200},
                    "history": [
                        {"role": "user", "content": "请记住编号 CTX-TEST。"},
                        {"role": "assistant", "content": "好的，记住 CTX-TEST。"},
                    ],
                    "prompt_max_tokens": 4000,
                },
                expect={"answer_all": ["ctx-test"], "compacted": True},
            )
        )
        config = MiniConfig(
            api_key="offline-test",
            api_base="https://example.invalid/v1",
        )
        provider = _SummaryAwareProvider("CTX-TEST")
        with mock.patch(
            "mini_agent.sdk.build_provider",
            lambda _config: provider,
        ):
            result = asyncio.run(run_case(case, config, EvalOptions(timeout=30)))
        self.assertEqual(result.error, "")
        self.assertTrue(result.answer_ok)
        self.assertTrue(result.compacted)
        self.assertTrue(result.compacted_ok)
        self.assertFalse(result.overflow)
        self.assertEqual(provider.summary_calls, 1)
        self.assertEqual(len(result.prompt_tokens), 1)

    def test_provider_transient_failure_recovers_through_retry(self) -> None:
        case = parse_case(
            _raw_case(
                id="rec-e2e",
                category="recovery",
                question="请回答 RETRY-OK",
                setup={"failure": "provider_transient"},
                expect={"answer_all": ["retry-ok"]},
            )
        )
        config = MiniConfig(
            api_key="offline-test",
            api_base="https://example.invalid/v1",
        )
        retrying = RetryingProvider(
            _StaticProvider("RETRY-OK"), max_attempts=3, base_delay=0.0
        )
        with mock.patch(
            "mini_agent.sdk.build_provider",
            lambda _config: retrying,
        ):
            result = asyncio.run(run_case(case, config, EvalOptions(timeout=30)))
        self.assertEqual(result.error, "")
        self.assertTrue(result.answer_ok)
        self.assertTrue(result.recovered)
        self.assertEqual(result.injected_failures, 1)


class RagDatasetTests(unittest.TestCase):
    def test_every_rag_case_is_retrievable_with_offline_embedder(self) -> None:
        cases = [case for case in load_cases() if case.category == "rag"]

        async def check() -> list[str]:
            failures: list[str] = []
            with tempfile.TemporaryDirectory() as temp:
                configure_knowledge_base(
                    HashingEmbedder(), index_dir=str(Path(temp) / "kb")
                )
                kb = get_knowledge_base()
                for case in cases:
                    await kb.clear()
                    for document in case.setup.knowledge:
                        await kb.ingest(
                            document["text"],
                            source=document["source"],
                            replace_source=True,
                        )
                    hits = await kb.search_hits(case.question, top_k=4)
                    payload = citation_payload(hits)
                    expected = case.expect.citations[0]
                    found = any(
                        (
                            item["source"] == expected.source
                            or Path(item["source"]).name == Path(expected.source).name
                        )
                        and expected.contains.casefold()
                        in f"{item['snippet']}\n{item['heading']}".casefold()
                        for item in payload
                    )
                    if not found:
                        failures.append(case.id)
                await kb.clear()
            return failures

        self.assertEqual(asyncio.run(check()), [])


class ReportTests(unittest.TestCase):
    def test_markdown_and_files(self) -> None:
        results = [
            CaseResult(
                case_id="qa-001",
                category="qa",
                answer="ok",
                answer_ok=True,
                ok=True,
            ),
            CaseResult(
                case_id="rag-001",
                category="rag",
                answer="bad",
                answer_ok=False,
                citation_ok=True,
                citation_expected=1,
                ok=False,
            ),
        ]
        report = {
            "generated_at": "2026-11-02T12:00:00+08:00",
            "model": "test-model",
            "api_base": "https://example.invalid/v1",
            "total_cases": 2,
            "metrics": aggregate(results, 8000),
            "by_category": aggregate_by_category(results, 8000),
            "cases": [item.as_dict() for item in results],
        }
        markdown = render_markdown(report)
        self.assertIn("回答正确率", markdown)
        self.assertIn("rag-001", markdown)
        self.assertIn("引用准确率", markdown)
        with tempfile.TemporaryDirectory() as temp:
            json_path, markdown_path = write_report(report, Path(temp))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_cases"], 2)


class SelectionTests(unittest.TestCase):
    def test_select_by_prefix_category_and_tag(self) -> None:
        cases = load_cases()
        self.assertTrue(all(case.category == "rag" for case in _select(cases, "rag", 0)))
        self.assertTrue(
            all(case.id.startswith("rec-00") for case in _select(cases, "rec-00", 0))
        )
        tagged = _select(cases, "retry", 0)
        self.assertTrue(tagged)
        self.assertTrue(all("retry" in case.tags for case in tagged))
        self.assertEqual(len(_select(cases, "", 3)), 3)


if __name__ == "__main__":
    unittest.main()