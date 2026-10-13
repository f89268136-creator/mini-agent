"""全链路离线 E2E 自测：用真实装配（MessageBus + SessionManager + AgentRunner +
AgentLoop + 自建 FakeChannel）跑通"渠道 → 入站队列 → 按会话分片 → ReAct →
落盘 → 出站路由 → 渠道"的完整链路，全程不联网、不依赖真实模型。

运行方式（Windows GBK 控制台建议加 PYTHONIOENCODING=utf-8 前缀）：

    python -m mini_agent.tests.test_pipeline_offline

覆盖场景：
    1) 总线回路 + 出站路由：publish_inbound 后渠道收到路由正确的回复；
    2) 同会话串行 + 顺序：同 sender 两条消息按序处理，模型侧最大并发为 1；
    3) 跨会话并行：确定性门闩法（不靠时间差）验证不同会话互不阻塞；
    4) JSONL 持久化 + 重启恢复：换一个 SessionManager 读同一目录还原历史；
    5) /reset 与 /help：命令不进模型不写历史；reset 后磁盘重写、水位线不串位；
    6) SDK 路径：MiniBot 构造后替换 provider，全离线走通 chat() 与落盘。

技巧：测试一律不自己 consume_outbound——loop 的出站分发器会消费出站队列并把
消息交给 channel.send()，测试自己再消费会与它抢消息。等回复用渠道侧的事件。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

from ..agent.loop import AgentLoop
from ..agent.runner import AgentRunner
from ..bus import InboundMessage, MessageBus, OutboundMessage
from ..channels import Channel
from ..config import MiniConfig
from ..providers import LLMProvider, LLMResponse, Message
from ..sdk import MiniBot
from ..session import SessionManager


class FakeChannel(Channel):
    """测试专用渠道：send() 只记录消息并唤醒等待者，不接触任何真实 IO。"""

    name = "fake"

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(bus)
        self.sent: list[OutboundMessage] = []
        self.updated = asyncio.Event()

    async def start(self) -> None:
        return None  # 测试渠道没有外部输入，start / stop 都不做事

    async def send(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)
        self.updated.set()

    async def stop(self) -> None:
        return None


class ScriptedProvider(LLMProvider):
    """E2E 用的假模型：按最后一条 user 消息生成 "答:xxx"。

    - delay：模拟模型耗时（验证同会话串行，最大并发应恒为 1）；
    - use_latch：跨会话并行的确定性门闩——“慢”先置慢开始事件再等放行，
      “快”等慢开始后放行。若系统把两个会话串行处理，慢会话会等 2 秒超时，
      回复变成“请求失败: TimeoutError”，断言当场失败（不靠时间差不靠猜）。
    """

    def __init__(self, delay: float = 0.0, use_latch: bool = False) -> None:
        self.delay = delay
        self.use_latch = use_latch
        self.calls = 0
        self.inflight = 0  # 当前正在处理中的请求数
        self.max_inflight = 0  # 观测到的峰值
        self.slow_started = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            text = ""
            for msg in reversed(messages):
                if msg.role == "user":
                    text = msg.content or ""
                    break
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.use_latch and "慢" in text:
                self.slow_started.set()
                await asyncio.wait_for(self.release.wait(), timeout=2)
                return LLMResponse(content="慢处理好", finish_reason="stop")
            if self.use_latch and "快" in text:
                await asyncio.wait_for(self.slow_started.wait(), timeout=2)
                self.release.set()
                return LLMResponse(content="快处理好", finish_reason="stop")
            return LLMResponse(content=f"答:{text}", finish_reason="stop")
        finally:
            self.inflight -= 1


def make_loop(tmpdir: str, provider: LLMProvider):
    """手工装配一套完整系统（不用 MiniBot，避免它构造带真实客户端的 provider）。"""
    bus = MessageBus()
    sessions = SessionManager(tmpdir, "你是测试助手")
    runner = AgentRunner(provider)
    channel = FakeChannel(bus)
    loop = AgentLoop(bus, runner, sessions, channels={"fake": channel})
    return bus, sessions, runner, channel, loop


async def wait_for_sent(channel: FakeChannel, count: int, timeout: float = 5.0) -> None:
    """等渠道累计收到 count 条回复；超时转成带中文信息的断言错误。"""

    async def _wait() -> None:
        while len(channel.sent) < count:
            channel.updated.clear()
            await channel.updated.wait()

    try:
        await asyncio.wait_for(_wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"等待渠道回复超时（{timeout} 秒），已收到 {len(channel.sent)} 条"
        ) from None


def read_lines(path: str) -> list[str]:
    """读 JSONL 文件的有效行（跳过空行）。"""
    with open(path, encoding="utf-8") as f:
        return [line for line in f if line.strip()]


async def test_bus_roundtrip() -> None:
    """场景 1：总线回路 + 出站路由（入站 -> 处理 -> 出站 -> 渠道）。"""
    tmpdir = tempfile.mkdtemp(prefix="mini_agent_bus_")
    try:
        bus, _, _, channel, loop = make_loop(tmpdir, ScriptedProvider())
        await loop.start()

        await bus.publish_inbound(InboundMessage(channel="fake", sender="alice", text="你好"))
        await wait_for_sent(channel, 1)

        reply = channel.sent[0]
        assert reply.text == "答:你好", f"回复内容不符: {reply.text}"
        assert reply.channel == "fake", f"回复渠道应为 fake，实际 {reply.channel}"
        assert reply.to == "alice", f"回复目标应为 alice，实际 {reply.to}"

        await loop.stop()
        assert loop.session_worker_count() == 0, "停机后不应再有活跃的会话 worker"

        print("[1] 总线回路 + 出站路由：通过")
        print("    回复:", reply.text, "| 渠道:", reply.channel, "| 目标:", reply.to)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_same_session_serial() -> None:
    """场景 2：同会话串行——两条消息按序处理，模型侧最大并发为 1。"""
    tmpdir = tempfile.mkdtemp(prefix="mini_agent_serial_")
    try:
        provider = ScriptedProvider(delay=0.03)  # 每条消息让模型“思考” 30 毫秒
        bus, _, _, channel, loop = make_loop(tmpdir, provider)
        await loop.start()

        # 连续发两条同 sender 的消息（都进 fake:alice 的会话队列）
        await bus.publish_inbound(InboundMessage(channel="fake", sender="alice", text="第一问"))
        await bus.publish_inbound(InboundMessage(channel="fake", sender="alice", text="第二问"))
        await wait_for_sent(channel, 2)

        texts = [m.text for m in channel.sent]
        assert texts == ["答:第一问", "答:第二问"], f"同会话回复顺序错乱: {texts}"
        assert provider.max_inflight == 1, (
            f"同会话应串行（模型侧最大并发 1），实际峰值 {provider.max_inflight}"
        )
        assert provider.calls == 2, f"两条消息应各调一次模型，实际 {provider.calls} 次"

        await loop.stop()
        assert loop.session_worker_count() == 0

        print(f"[2] 同会话串行 + 顺序：通过（max_inflight={provider.max_inflight}，按序回复）")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_cross_session_parallel() -> None:
    """场景 3：跨会话并行——确定性门闩法，慢会话不阻塞快会话。"""
    tmpdir = tempfile.mkdtemp(prefix="mini_agent_parallel_")
    try:
        provider = ScriptedProvider(use_latch=True)
        bus, _, _, channel, loop = make_loop(tmpdir, provider)
        await loop.start()

        # 先发“慢”（fake:slow，会挂起等放行），再发“快”（fake:fast，负责放行）。
        # 两个会话各有自己的 worker：慢会话挂起时，快会话立即开始处理。
        await bus.publish_inbound(InboundMessage(channel="fake", sender="slow", text="慢"))
        await bus.publish_inbound(InboundMessage(channel="fake", sender="fast", text="快"))
        await wait_for_sent(channel, 2)

        texts = sorted(m.text for m in channel.sent)
        assert texts == ["快处理好", "慢处理好"], (
            f"跨会话并行失败（若全局串行，慢会话会 2 秒超时报 TimeoutError）: {texts}"
        )
        assert loop.session_worker_count() == 2, (
            f"两个会话应各有独立 worker，实际 {loop.session_worker_count()} 个"
        )

        await loop.stop()
        assert loop.session_worker_count() == 0

        print("[3] 跨会话并行（门闩法）：通过（两会话各一 worker，互不阻塞）")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_jsonl_persistence_and_restore() -> None:
    """场景 4：一轮对话落盘为 JSONL；换一个 SessionManager 能恢复历史。"""
    tmpdir = tempfile.mkdtemp(prefix="mini_agent_persist_")
    try:
        bus, _, _, channel, loop = make_loop(tmpdir, ScriptedProvider())
        await loop.start()

        await bus.publish_inbound(
            InboundMessage(channel="fake", sender="alice", text="记住：我喜欢苹果")
        )
        await wait_for_sent(channel, 1)
        assert channel.sent[0].text == "答:记住：我喜欢苹果"

        await loop.stop()
        assert loop.session_worker_count() == 0

        # 会话 key "fake:alice" -> 文件名清洗为 fake_alice.jsonl
        path = os.path.join(tmpdir, "fake_alice.jsonl")
        assert os.path.exists(path), f"会话文件未落盘: {path}"
        lines = read_lines(path)
        assert len(lines) == 3, (
            f"一轮对话应落盘 3 行（system/user/assistant），实际 {len(lines)}"
        )
        assert json.loads(lines[0])["role"] == "system", "首行应为 system"
        assert "记住：我喜欢苹果" in lines[1], "user 原文应可读地落在 JSONL 里（ensure_ascii=False）"

        # 重启：新的 SessionManager 读同一目录（同 system_prompt）
        sm2 = SessionManager(tmpdir, "你是测试助手")
        session2 = sm2.get_or_create("fake:alice")
        roles = [m.role for m in session2.messages]
        assert roles == ["system", "user", "assistant"], f"恢复出的消息结构不符: {roles}"
        assert session2.messages[0].content == "你是测试助手", "system 人设未恢复"
        assert session2.messages[1].content == "记住：我喜欢苹果", "user 消息原文未恢复"
        assert session2.messages[2].content == "答:记住：我喜欢苹果", "assistant 回复未恢复"

        print("[4] JSONL 持久化 + 重启恢复：通过（fake_alice.jsonl，3 行，历史完整）")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_reset_and_help_commands() -> None:
    """场景 5：/help 与 /reset 不进模型、不写历史；reset 后水位线重排不串位。"""
    tmpdir = tempfile.mkdtemp(prefix="mini_agent_reset_")
    try:
        provider = ScriptedProvider()
        bus, _, _, channel, loop = make_loop(tmpdir, provider)
        await loop.start()
        path = os.path.join(tmpdir, "fake_alice.jsonl")

        # 正常聊一轮：文件 3 行（system/user/assistant）
        await bus.publish_inbound(InboundMessage(channel="fake", sender="alice", text="第一轮"))
        await wait_for_sent(channel, 1)
        assert len(read_lines(path)) == 3, "一轮对话后应有 3 行"

        # /help：不进模型（calls 不增）、不写历史（文件仍是 3 行）
        await bus.publish_inbound(InboundMessage(channel="fake", sender="alice", text="/help"))
        await wait_for_sent(channel, 2)
        assert "/reset" in channel.sent[1].text, f"/help 应返回命令说明: {channel.sent[1].text}"
        assert provider.calls == 1, "/help 不应调用模型"
        assert len(read_lines(path)) == 3, "/help 不应写历史"

        # /reset：清空会话并立刻重写文件（只剩 system 1 行）
        await bus.publish_inbound(InboundMessage(channel="fake", sender="alice", text="/reset"))
        await wait_for_sent(channel, 3)
        assert channel.sent[2].text == "（对话已清空）", f"/reset 回复不符: {channel.sent[2].text}"
        assert provider.calls == 1, "/reset 不应调用模型"
        lines = read_lines(path)
        assert len(lines) == 1, f"/reset 后应只剩 system 一行，实际 {len(lines)}"
        assert json.loads(lines[0])["role"] == "system", "/reset 后保留的应是 system"

        # 清空后再问一轮：水位线已重置，增量 append 应恰好再写 2 行（共 3 行）
        await bus.publish_inbound(
            InboundMessage(channel="fake", sender="alice", text="清空后再问")
        )
        await wait_for_sent(channel, 4)
        assert channel.sent[3].text == "答:清空后再问", f"清空后回答不符: {channel.sent[3].text}"
        lines = read_lines(path)
        assert len(lines) == 3, f"/reset 后新一轮应共 3 行（1+2），实际 {len(lines)}"
        roles = [json.loads(line)["role"] for line in lines]
        assert roles == ["system", "user", "assistant"], f"行结构不符: {roles}"
        assert all("/reset" not in line and "/help" not in line for line in lines), (
            "控制命令不应写进会话历史"
        )

        await loop.stop()
        assert loop.session_worker_count() == 0

        print("[5] /help 与 /reset：通过（命令不进模型不写历史；reset 后水位线正确，共 3 行）")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_sdk_path() -> None:
    """场景 6：SDK 路径——MiniBot 换 FakeProvider 后全离线走通 chat() 与落盘。"""
    tmpdir = tempfile.mkdtemp(prefix="mini_agent_sdk_")
    try:
        # MiniBot 构造时会创建真实的 AsyncOpenAI 客户端（不发请求、不需要网络），
        # 拿到实例后立刻换成 FakeProvider，之后所有 chat 都是全离线。
        bot = MiniBot(MiniConfig(channels=[], sessions_dir=tmpdir))
        fake = ScriptedProvider()
        bot.runner.provider = fake

        answer = await bot.chat("你好", session_key="sdk:test")
        assert answer == "答:你好", f"SDK 回答不符: {answer}"
        assert fake.calls == 1, f"SDK 首轮应只调一次模型，实际 {fake.calls}"

        answer2 = await bot.chat("第二轮", session_key="sdk:test")
        assert answer2 == "答:第二轮", f"SDK 第二轮回答不符: {answer2}"

        # 会话 key "sdk:test" -> 文件名清洗为 sdk_test.jsonl
        path = os.path.join(tmpdir, "sdk_test.jsonl")
        assert os.path.exists(path), f"SDK 会话未落盘: {path}"
        lines = read_lines(path)
        assert len(lines) == 5, (
            f"两轮对话应落盘 5 行（system + 两对问答），实际 {len(lines)}"
        )
        roles = [json.loads(line)["role"] for line in lines]
        assert roles == ["system", "user", "assistant", "user", "assistant"], f"行结构不符: {roles}"

        # SDK 直调 handle_text，不经过总线与渠道，自然也没有会话 worker
        assert bot.channels == {}, "channels=[] 时不应装配任何渠道"
        assert bot.loop.session_worker_count() == 0, "SDK 路径不应产生会话 worker"

        print("[6] SDK 路径：通过（回答:", answer, "，文件: sdk_test.jsonl，5 行）")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _amain() -> None:
    await test_bus_roundtrip()
    await test_same_session_serial()
    await test_cross_session_parallel()
    await test_jsonl_persistence_and_restore()
    await test_reset_and_help_commands()
    await test_sdk_path()
    print("pipeline 离线自测通过（总线 / 分片 / 持久化 / 命令 / SDK 全链路 E2E，全程离线）")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
