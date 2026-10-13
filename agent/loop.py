"""部件 7（编排层）：AgentLoop —— 把"总线 / 会话 / 渠道 / 引擎"装配成完整系统。

对应 nanobot/agent/loop.py 的最小版。消息的完整流转：

    渠道 ──publish──► MessageBus（inbound 队列）
                            │ consume（唯一消费者：_inbound_dispatcher）
                            ▼
             按 session_key 找到该会话专属 queue + worker
                            │
                            ▼
             worker：handle_inbound → runner.run（ReAct）
                            │
                            ▼
             OutboundMessage ──publish──► MessageBus（outbound 队列）
                            │ consume（_outbound_dispatcher）
                            ▼
                          渠道.send

核心机制——"按会话分片"（正是 nanobot 的分片模型）：
    每条会话有自己的一条 asyncio.Queue 和一个专属 worker 任务。
      - 同一会话的消息排在同一条队列里 → 按顺序串行处理，
        不会出现"上一条还没答完，下一条就插进来"的历史错乱；
      - 不同会话各用各的 worker → 天然并行，
        慢会话不拖累快会话（事件循环在 await 让出时切换）。

教学版刻意简化（对照 nanobot）：
    - 没有钩子（hooks）、准入闸门、命令路由（command router）、上下文压缩等；
    - 会话 worker 常驻不回收（nanobot 有 TTL 清理与弱引用锁）；
    - 渠道不存在时只打一条中文警告，生产版还会做重试与投递确认。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..bus import InboundMessage, MessageBus, OutboundMessage
from ..providers.base import Message
from ..session import SessionManager
from .runner import AgentRunner

if TYPE_CHECKING:  # 仅用于类型标注，避免运行时引入渠道包的额外依赖
    from ..channels import Channel


class AgentLoop:
    def __init__(
        self,
        bus: MessageBus,
        runner: AgentRunner,
        sessions: SessionManager,
        channels: dict[str, Channel] | None = None,
    ):
        # 四个部件都由外部装配好后传进来（由 sdk.MiniBot 完成总装），
        # loop 自己不 new 任何东西——"接住零件"，便于测试时逐个替换。
        self.bus = bus
        self.runner = runner
        self.sessions = sessions
        self.channels: dict[str, Channel] = channels or {}

        # 分片状态：会话键 → 该会话专属的入站队列；会话键 → worker 任务。
        # 这是跨回合（跨消息）的状态，所以挂在实例上；单条消息的中间状态
        # 只活在 handle_inbound 的局部变量里，不升级成实例属性。
        self._session_workers: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}

        # 后台分发任务的引用：不保存引用的话，事件循环可能在任务还没跑完时
        # 把它当垃圾回收掉（asyncio 的经典坑）。
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    # ------------------------------------------------------------------
    # 单条消息处理（SDK 直调与 worker 共用同一条逻辑）
    # ------------------------------------------------------------------
    async def handle_inbound(self, inbound: InboundMessage) -> OutboundMessage:
        """处理一条入站消息，返回要发回渠道的出站消息。

        这是"总线路径"的入口：worker 从会话队列取出消息后调用它；
        它只做一次转发——把消息里的渠道信息拆开，交给 handle_text 统一处理。
        """
        return await self.handle_text(
            channel=inbound.channel,
            sender=inbound.sender,
            text=inbound.text,
            session_key=inbound.session_key,  # 渠道侧算好的会话键，必须显式透传
        )

    async def handle_text(
        self,
        channel: str,
        sender: str,
        text: str,
        session_key: str | None = None,
    ) -> OutboundMessage:
        """通道/发送者/文本 → 出站消息 的统一处理逻辑（命令层 + 会话 + ReAct + 持久化）。

        为什么单独拆出来：InboundMessage.session_key 是"渠道+发送者"自动生成的
        property（不可改），而 SDK 需要自定义会话键；所以把核心逻辑做成接受
        session_key 参数的函数——缺省时同样退化为 f"{channel}:{sender}"，
        与 InboundMessage.session_key 的算法保持一致。
        """
        key = session_key or f"{channel}:{sender}"
        command = text.strip()

        # ---- 命令层：以 / 开头的控制命令不进模型、不进历史 ----
        if command == "/reset":
            # 清空会话：reset() 只留 system 提示词，然后立刻持久化，防止进程退出丢状态。
            session = self.sessions.get_or_create(key)
            session.reset()
            await self.sessions.save(session)
            return OutboundMessage(channel=channel, to=sender, text="（对话已清空）")
        if command == "/help":
            # 命令清单（教学版只提供最基础的几条；/exit 由 CLI 渠道自己处理，
            # 因为它要把"退出"翻译成渠道侧的 closed 事件，而不是一条回复）。
            return OutboundMessage(
                channel=channel,
                to=sender,
                text="/reset 清空当前会话；/exit 退出程序；其他内容会直接交给模型。",
            )

        # ---- 正常消息：取会话 → 加用户消息 → 跑 ReAct → 存回答 ----
        session = self.sessions.get_or_create(key)
        session.add_message(Message(role="user", content=text))

        # get_history() 不加参数：历史截断上限（history_limit）是 SessionManager 的
        # 职责——它创建 Session 时就把上限注入进去，loop 不需要也不应该再管。
        # 外层套 list()：拿一份快照副本给 runner，runner 回灌的 assistant/tool
        # 消息只留在快照上，不会污染会话存储（生产版 nanobot 会把整个回合的
        # transcript 都写回会话，这里教学版只存"用户问 + 最终答"两条，刻意简化）。
        history = list(session.get_history())

        answer = await self.runner.run(history)

        session.add_message(Message(role="assistant", content=answer))
        await self.sessions.save(session)
        return OutboundMessage(channel=channel, to=sender, text=answer)

    # ------------------------------------------------------------------
    # 后台分发：入站分片 + 出站路由
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动两个常驻后台任务：入站分发器、出站分发器。"""
        if self._running:
            return  # 幂等：重复 start 不重复起任务
        self._running = True
        self._tasks = [
            asyncio.create_task(self._inbound_dispatcher(), name="mini-agent-inbound"),
            asyncio.create_task(self._outbound_dispatcher(), name="mini-agent-outbound"),
        ]

    async def _inbound_dispatcher(self) -> None:
        """总入口：从总线取消息，按 session_key 分片到会话队列。

        它自己是"唯一消费者"，但只做分发不做处理，所以任何一条会话处理得慢，
        都不会卡住分发——这正是 nanobot 在总线与 worker 之间加一层队列的原因。
        """
        while True:
            inbound = await self.bus.consume_inbound()

            # 首次见到这个会话键 → 现场建队列 + 专属 worker（单任务里执行，
            # 不存在两个消息同时创建同一会话 worker 的竞态）。
            queue = self._session_workers.get(inbound.session_key)
            if queue is None:
                queue = asyncio.Queue()
                self._session_workers[inbound.session_key] = queue
                self._worker_tasks[inbound.session_key] = asyncio.create_task(
                    self._session_worker(inbound.session_key, queue),
                    name=f"mini-agent-session:{inbound.session_key}",
                )

            # 入队即返回：同一会话的多条消息在这里排队，由 worker 一条条处理。
            await queue.put(inbound)

    async def _session_worker(self, session_key: str, queue: asyncio.Queue[InboundMessage]) -> None:
        """会话 worker：串行消费自己的队列，处理完把回复发回总线。

        每个会话一个这样的任务实例：
          - 同会话串行：while True 逐条 await handle_inbound，上一条处理完才取下一条；
          - 跨会话并行：不同 worker 是不同任务，各自 await 时事件循环自由切换。
        """
        while True:
            inbound = await queue.get()
            try:
                outbound = await self.handle_inbound(inbound)
            except asyncio.CancelledError:
                # 停机信号：原样抛出，让 asyncio.gather 能正常收尾
                raise
            except Exception as exc:
                # 单条消息失败不能把 worker 打死（否则整个会话从此失联），
                # 转成一条中文错误回复发回去——与旧版 REPL 的兜底文案一致。
                outbound = OutboundMessage(
                    channel=inbound.channel,
                    to=inbound.sender,
                    text=f"请求失败: {type(exc).__name__}: {exc}",
                )
            try:
                await self.bus.publish_outbound(outbound)
            finally:
                queue.task_done()

    async def _outbound_dispatcher(self) -> None:
        """出站路由：从总线取出回复，交给对应渠道发送。"""
        while True:
            msg = await self.bus.consume_outbound()
            channel = self.channels.get(msg.channel)
            if channel is None:
                # 渠道没装配（配置里没选它）或名字拼错——丢弃但留痕，便于排查
                print(f"[mini_agent] 警告：没有名为 {msg.channel!r} 的渠道，回复被丢弃：{msg.text}")
                continue
            try:
                await channel.send(msg)
            except Exception as exc:
                # 渠道发送失败不影响其他渠道与其他消息
                print(f"[mini_agent] 警告：渠道 {msg.channel!r} 发送失败: {type(exc).__name__}: {exc}")

    async def stop(self) -> None:
        """停机：取消全部分发任务与会话 worker，并等它们善后完成。"""
        tasks = [
            task
            for task in (*self._tasks, *self._worker_tasks.values())
            if not task.done()
        ]
        self._tasks.clear()
        self._worker_tasks.clear()
        self._session_workers.clear()

        for task in tasks:
            task.cancel()
        if tasks:
            # return_exceptions=True：CancelledError 属于预期结果，不当错误抛出去
            await asyncio.gather(*tasks, return_exceptions=True)
        self._running = False

    # ------------------------------------------------------------------
    # 教学观测
    # ------------------------------------------------------------------
    def session_worker_count(self) -> int:
        """当前活跃的会话 worker 数（= 已经建过分片的会话数）。

        用它直观观察分片机制：两个会话各发一条消息后，这里会变成 2；
        慢会话处理期间，快会话依然能立刻拿到自己的 worker 响应。
        """
        return sum(1 for task in self._worker_tasks.values() if not task.done())
