"""命令行入口：python -m mini_agent（也可从项目根 run.py 点绿三角启动）。

main() 保持"同步入口"的签名（run.py 依赖它），内部用 asyncio.run 驱动异步系统：

    load_config → 装配 MiniBot → 打印启动横幅 → 启动（loop + 各渠道）
    → 等退出信号（CLI 渠道的 closed 事件；无 CLI 就等 Ctrl+C）
    → 优雅停机（先渠道后 loop）→ 打印"已退出"

命令提示：/reset 清空当前会话；/exit 退出程序。
"""

from __future__ import annotations

import asyncio

from .sdk import MiniBot


async def _amain() -> None:
    # 1) 装配：配置来自 load_config()（环境变量 + 配置文件），再交给门面总装
    bot = MiniBot.from_config()

    # 2) 启动横幅：一眼看清"连的是哪个模型、开了哪些渠道、会话历史存在哪"
    channels_text = ", ".join(bot.config.channels) if bot.config.channels else "(无)"
    print("=" * 60)
    print("mini_agent 已启动（输入 /exit 退出，/reset 清空对话）")
    print(f"  模型:     {bot.config.model}")
    print(f"  渠道:     {channels_text}")
    print(f"  会话目录: {bot.config.sessions_dir}")
    print("=" * 60)

    try:
        await bot.start()

        # 3) 等退出信号。
        #    cli 渠道：用户输入 /exit → CliChannel 把 closed 事件置位，等它就够了；
        #    没有 cli（如纯 echo 模式）：没有"用户说退出"的通道，只能等 Ctrl+C，
        #    用一个永不 set 的 Event 挂着即可。
        cli = bot.loop.channels.get("cli")
        closed = getattr(cli, "closed", None) if cli is not None else None
        if closed is None:
            print("（当前没有 CLI 渠道，按 Ctrl+C 退出）")
            closed = asyncio.Event()
        await closed.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C：CPython 常把 SIGINT 变成对主任务的取消（CancelledError），
        # 也可能直接在这里抛出 KeyboardInterrupt——两种都视为正常退出信号，
        # 统一交给 finally 走善后流程。
        pass
    finally:
        await bot.aclose()
    print("已退出")


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        # 某些平台上 Ctrl+C 会从 asyncio.run 的收尾阶段才抛出；
        # 此时 _amain 内已完成善后（aclose 与提示），这里静默收工。
        pass


if __name__ == "__main__":
    main()
