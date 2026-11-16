"""可选 Langfuse 追踪：未启用时不导入 langfuse、不修改 openai 客户端。

设计约束：
- 默认关闭。没有显式配置时，业务代码只付出一次函数调用的成本；
- 密钥只从 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 读取，不接受配置字段；
- 启用时 ``import langfuse.openai`` 会全局包装 openai 的 create 调用，
  因此 provider 与嵌入模型都无需改动；
- 追踪失败不能影响对话，flush/shutdown 只记录日志。
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_client: Any = None

# 评测包用于单点评委：工具结果与记忆块都可能很长，必须限长，
# 完整内容仍然保留在各自的 tool.* span 与提示词里。
_MAX_TOOL_RECORDS = 20
_MAX_TOOL_RESULT_CHARS = 2_000
_MAX_TOOL_ARGUMENT_CHARS = 1_000
_MAX_MEMORY_CHARS = 8_000
_MAX_TODO_CHARS = 4_000
_MAX_HISTORY_EXCERPT_CHARS = 4_000
_MAX_HISTORY_ITEM_CHARS = 200


@dataclass
class _TurnRecorder:
    question: str = ""
    answer: str | None = None
    memory_enabled: bool = False
    memory_context: str = ""
    todo_enabled: bool = False
    todo_context: str = ""
    history_messages: int = 0
    prompt_messages: int = 0
    prompt_chars: int = 0
    history_excerpt: str = ""
    iterations: int = 0
    tools: list[dict[str, Any]] = field(default_factory=list)


_recorder: ContextVar[_TurnRecorder | None] = ContextVar(
    "mini_agent_turn_recorder", default=None
)


def _clip_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…（已截断）"


def _clip_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, (dict, list)):
        return _clip_text(str(arguments), _MAX_TOOL_ARGUMENT_CHARS)
    try:
        rendered = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return _clip_text(str(arguments), _MAX_TOOL_ARGUMENT_CHARS)
    return arguments if len(rendered) <= _MAX_TOOL_ARGUMENT_CHARS else _clip_text(
        rendered, _MAX_TOOL_ARGUMENT_CHARS
    )


def record_memory_context(text: str, *, enabled: bool) -> None:
    """记录本轮注入的长期记忆；未启用追踪时无操作。"""
    recorder = _recorder.get()
    if recorder is not None:
        recorder.memory_enabled = enabled
        recorder.memory_context = _clip_text(text, _MAX_MEMORY_CHARS)


def record_todo_context(text: str, *, enabled: bool) -> None:
    """记录本轮注入的任务清单；未启用追踪时无操作。"""
    recorder = _recorder.get()
    if recorder is not None:
        recorder.todo_enabled = enabled
        recorder.todo_context = _clip_text(text, _MAX_TODO_CHARS)


def render_history_excerpt(messages: list[Any]) -> str:
    """把实际发给模型的用户消息渲染成原文证据，供评委核对回答依据。

    只取 user 消息：身份、偏好等事实的原始来源都在这里，且体量有限；
    模型回答已由 eval.packet 的 agent_output 提供，工具结果在 tool_records 里。
    传入的应当是裁剪后的 prompt，而不是完整 history——否则评委可能看到
    模型根本没读到的旧消息，把"无依据"误判成"有依据"。
    """
    lines: list[str] = []
    for message in messages:
        if getattr(message, "role", "") != "user":
            continue
        text = (getattr(message, "content", "") or "").strip()
        if text:
            lines.append(f"user: {_clip_text(text, _MAX_HISTORY_ITEM_CHARS)}")
    # 超预算时保留最早的几条：身份/偏好通常在会话早期出现，当前提问另有
    # user_question 字段，丢掉末尾不会损失关键证据。
    return _clip_text("\n".join(lines), _MAX_HISTORY_EXCERPT_CHARS)


def record_turn_context(
    *,
    history_messages: int,
    prompt_messages: int,
    prompt_chars: int,
    history_excerpt: str = "",
) -> None:
    recorder = _recorder.get()
    if recorder is not None:
        recorder.history_messages = history_messages
        recorder.prompt_messages = prompt_messages
        recorder.prompt_chars = prompt_chars
        recorder.history_excerpt = history_excerpt


def record_iteration() -> None:
    recorder = _recorder.get()
    if recorder is not None:
        recorder.iterations += 1


def record_tool_call(
    *,
    name: str,
    arguments: Any,
    result: str,
    ok: bool,
    duration_ms: float,
) -> None:
    recorder = _recorder.get()
    if recorder is None or len(recorder.tools) >= _MAX_TOOL_RECORDS:
        return
    recorder.tools.append(
        {
            "name": name,
            "arguments": _clip_arguments(arguments),
            "ok": ok,
            "duration_ms": round(duration_ms, 1),
            "result": _clip_text(result, _MAX_TOOL_RESULT_CHARS),
        }
    )


def record_answer(answer: str) -> None:
    recorder = _recorder.get()
    if recorder is not None:
        recorder.answer = answer


@contextmanager
def suppress_turn_recording() -> Iterator[None]:
    """子代理的工具与记忆不并入父回合的评测包。"""
    token = _recorder.set(None)
    try:
        yield
    finally:
        _recorder.reset(token)


class _SafeSpan:
    """包住 Langfuse span：未启用或更新失败都不影响业务代码。"""

    def __init__(self, span: Any = None):
        self._span = span

    def update(self, **kwargs: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.update(**kwargs)
        except Exception:
            logger.exception("Langfuse span 更新失败")


_NOOP = _SafeSpan()


def _start(**kwargs: Any) -> Any:
    try:
        return _client.start_as_current_observation(**kwargs)
    except Exception:
        logger.exception("创建 Langfuse span 失败，已跳过追踪")
        return None


def _close(manager: Any, exc_info: tuple) -> None:
    try:
        manager.__exit__(*exc_info)
    except Exception:
        logger.exception("关闭 Langfuse span 失败")


@contextmanager
def _managed(manager: Any, label: str) -> Iterator[_SafeSpan]:
    """进入第三方上下文管理器，失败时降级为无操作，但保留业务异常。"""
    try:
        span = manager.__enter__()
    except Exception:
        logger.exception("Langfuse %s 创建失败，已跳过追踪", label)
        yield _NOOP
        return
    try:
        yield _SafeSpan(span)
    except BaseException:
        _close(manager, sys.exc_info())
        raise
    else:
        _close(manager, (None, None, None))


_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|password|secret|token|cookie)", re.IGNORECASE
)
# 默认只上报结构（工具名、状态、耗时、计数），不上报正文。
_CONTENT_KEYS = {
    "input",
    "output",
    "text",
    "content",
    "messages",
    "prompt",
    "response",
    "question",
    "answer",
    "arguments",
    "result",
    "history_excerpt",
    "agent_output",
    "user_question",
    "memory_context",
    "todo_context",
}
_REDACTED = "[已脱敏]"


def redact_trace_payload(data: Any) -> Any:
    """递归脱敏 Langfuse 上报内容：密钥类键清空，正文类键替换为占位符。

    Langfuse 的 ``mask`` 回调要求返回同结构字典；保留工具名、状态、
    耗时与计数，便于排查问题，同时避免把用户文档和对话内容传出本机。
    """
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            text_key = str(key)
            if _SENSITIVE_KEY.search(text_key):
                cleaned[key] = _REDACTED
            elif text_key.lower() in _CONTENT_KEYS:
                cleaned[key] = _REDACTED
            else:
                cleaned[key] = redact_trace_payload(value)
        return cleaned
    if isinstance(data, list):
        return [redact_trace_payload(item) for item in data]
    return data


def configure_tracing(
    *,
    enabled: bool = False,
    base_url: str | None = None,
    sample_rate: float | None = None,
    mask: Any = None,
) -> None:
    """初始化（或关闭）进程级 Langfuse 客户端。

    重复调用时旧客户端不会自动 shutdown；应用层应在退出前调用
    :func:`shutdown_tracing`。
    """
    global _client
    if not enabled:
        _client = None
        return
    try:
        from langfuse import Langfuse
        import langfuse.openai  # noqa: F401  全局包装 openai 调用
    except ImportError as exc:
        raise RuntimeError(
            "已启用 Langfuse，但未安装依赖：pip install 'langfuse>=5,<6'"
        ) from exc

    _client = Langfuse(
        base_url=base_url or None,
        sample_rate=sample_rate,
        mask=mask,
        # public_key / secret_key 故意不传：只允许从环境变量读取
    )


def _emit_eval_packet(recorder: _TurnRecorder) -> None:
    """把本轮评测要素写成 eval.packet 子观察，供单点评委直接读取。

    根 span 只带输入文本与最终回答；工具记录、记忆块等分散在子 span 或
    根本没上报。单点评委无法访问兄弟观察，因此这里聚合一份完整快照。
    """
    if _client is None:
        return
    payload_input = {
        "user_question": recorder.question,
        "memory_enabled": recorder.memory_enabled,
        "memory_context": recorder.memory_context,
        "todo_enabled": recorder.todo_enabled,
        "todo_context": recorder.todo_context,
        "tool_records": recorder.tools,
        "history_messages": recorder.history_messages,
        "prompt_messages": recorder.prompt_messages,
        "prompt_chars": recorder.prompt_chars,
        "history_excerpt": recorder.history_excerpt,
    }
    payload_output = {
        "agent_output": recorder.answer,
        "iterations": recorder.iterations,
        "tool_calls": len(recorder.tools),
        "tool_errors": sum(1 for item in recorder.tools if not item["ok"]),
    }
    with observation(
        "eval.packet",
        as_type="span",
        input=payload_input,
        metadata={"purpose": "agent_evaluation"},
    ) as span:
        span.update(output=payload_output)


@contextmanager
def turn_span(
    *, session_key: str, user_id: str, channel: str, text: str
) -> Iterator[Any]:
    """一次用户回合的根 span，同时设置 session/user/tags。"""
    if _client is None:
        yield _NOOP
        return
    try:
        from langfuse import propagate_attributes
    except ImportError:
        logger.exception("当前 Langfuse 版本不支持 propagate_attributes，已跳过追踪")
        yield _NOOP
        return

    try:
        attributes = propagate_attributes(
            session_id=session_key,
            user_id=user_id,
            tags=["mini_agent", channel],
            trace_name="agent.turn",
        )
    except Exception:
        logger.exception("创建 Langfuse 属性传播失败，已跳过追踪")
        yield _NOOP
        return

    manager = _start(
        as_type="span",
        name="agent.turn",
        input={"channel": channel, "text": text},
    )
    if manager is None:
        yield _NOOP
        return
    with _managed(attributes, "turn attributes"):
        with _managed(manager, "agent.turn") as span:
            recorder = _TurnRecorder(question=text)
            token = _recorder.set(recorder)
            try:
                yield span
            finally:
                try:
                    _emit_eval_packet(recorder)
                except Exception:
                    logger.exception("写入 Langfuse 评测包失败")
                _recorder.reset(token)


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """通用子 span：ReAct 轮次、工具、子代理都用它。"""
    if _client is None:
        yield _NOOP
        return
    manager = _start(
        as_type=as_type, name=name, input=input, metadata=metadata
    )
    if manager is None:
        yield _NOOP
        return
    with _managed(manager, name) as span:
        yield span


def shutdown_tracing() -> None:
    """关闭后台线程并刷新缓冲；进程退出前调用。"""
    global _client
    if _client is None:
        return
    try:
        _client.shutdown()
    except Exception:
        logger.exception("Langfuse shutdown 失败")
    finally:
        _client = None
