"""部件 8（编排层）：MiniBot —— 对外唯一门面（对应 nanobot/nanobot.py 的最小版）。

职责：把散落的零件"装配"成一个能跑的系统，外部只需认识这一个类：

    MiniBot(config)
        ├── MessageBus()                     # 渠道 ↔ 内核 解耦的消息总线
        ├── SessionManager(...)              # 按会话键存取历史的仓库
        ├── AgentRunner(build_provider(...)) # ReAct 引擎（含 LLM provider）
        └── AgentLoop(bus, runner, sessions, channels=build_channels(...))

用法（一行示例）：
    answer = await MiniBot.from_config().chat("你好")

装配本身是同步的：构造 MiniBot 不会建立任何网络连接、不会起后台任务，
真正的资源（LLM 客户端、渠道监听）在 start() 或第一次 chat() 时才被使用。
"""

from __future__ import annotations

from .agent.loop import AgentLoop
from .agent.runner import AgentRunner
from .bus import MessageBus
from .channels import Channel, build_channels
from .config import MiniConfig, load_config
from .providers import build_provider
from .session import SessionManager


class MiniBot:
    def __init__(self, config: MiniConfig | None = None):
        # 1) 配置：不传就用 load_config() 走默认来源（环境变量 / 配置文件）
        self.config: MiniConfig = config or load_config()

        # 2) 总线：两个方向的队列（inbound：渠道→内核；outbound：内核→渠道）
        self.bus = MessageBus()

        # 3) 会话仓库：负责历史截断（history_limit）与落盘（sessions_dir）
        self.sessions = SessionManager(
            self.config.sessions_dir,
            self.config.system_prompt,
            self.config.history_limit,
        )

        # 4) 引擎：provider 从配置构造（模型名 / 密钥 / 端点），runner 只管循环
        self.runner = AgentRunner(
            build_provider(self.config),
            self.config.max_iterations,
        )

        # 5) 渠道：按配置里的名字列表装配（"cli" → CliChannel，"echo" → EchoChannel）
        self.channels: dict[str, Channel] = build_channels(
            self.config.channels,
            self.bus,
            echo_port=self.config.echo_port,
        )

        # 6) 总装：loop 把上面所有部件串起来（接住零件，而不是自己造零件）
        self.loop = AgentLoop(
            self.bus,
            self.runner,
            self.sessions,
            channels=self.channels,
        )

    @classmethod
    def from_config(cls, path: str | None = None) -> "MiniBot":
        """从配置文件（缺省时用默认查找路径）构造——开箱即用的唯一入口。"""
        return cls(load_config(path))

    # ------------------------------------------------------------------
    # 直接对话（不走渠道/总线，SDK 场景：脚本、测试、嵌入到别的程序）
    # ------------------------------------------------------------------
    async def chat(self, text: str, session_key: str = "sdk:local") -> str:
        """直接问一句，返回回答文本；历史仍按 session_key 正常存取与持久化。

        InboundMessage.session_key 是由 channel+sender 自动生成的 property，
        没法直接指定"我想用哪个会话"；所以这里绕开 InboundMessage，
        直调 loop.handle_text 并显式传 session_key——这也是 handle_text
        从 handle_inbound 里拆出来的原因。
        """
        outbound = await self.loop.handle_text(
            channel="sdk",
            sender="local",
            text=text,
            session_key=session_key,
        )
        return outbound.text

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动整个系统。

        顺序：先起 loop（消费者）再起各渠道（生产者）。
        理由：渠道一启动就可能立刻收到用户输入并 publish 进总线，
        此时 loop 的分发任务必须已经在等着消费，否则消息只会静静躺在
        队列里（虽然不会丢，但启动体验和排查都更差）；先 loop 更稳。
        """
        await self.loop.start()
        for channel in self.channels.values():
            await channel.start()

    async def stop(self) -> None:
        """停机，顺序与 start 相反：先停渠道（生产者）再停 loop（消费者），
        让总线里已经排队的消息尽量先被处理完，再收掉分发任务。"""
        for name, channel in self.channels.items():
            try:
                await channel.stop()
            except Exception as exc:
                # 单个渠道停不下来不能连累整体退出流程
                print(f"[mini_agent] 警告：停止渠道 {name!r} 出错: {type(exc).__name__}: {exc}")
        await self.loop.stop()

    async def aclose(self) -> None:
        """异步关闭（同 stop；提供这个别名是为了贴合 async with / aclose 的习惯）。"""
        await self.stop()

    async def __aenter__(self) -> "MiniBot":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
