"""命令行入口：``mini-agent``（也可 ``python -m mini_agent`` 或用 run.py）。

默认（无参数）是"本机应用式启动"：先做预检，再打开 Web 界面；
首次运行落在设置页，配置完成后直接进入对话。

子命令：
    无参数 / web   默认 Web 界面（未配置时打开设置页）
    chat           终端交互对话
    login          终端凭据向导（写入系统钥匙串）
    doctor         环境诊断
    init           生成最小配置文件

等价的参数形式：--web / --init / --doctor / -m / -c / --model / --base-url。
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import importlib.util
import json
import logging
import os
import signal
import sys
import threading
import webbrowser
from dataclasses import replace

# 直接运行本文件时不存在包上下文；把项目根加入 sys.path 后统一走绝对导入。
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_agent import __version__
from mini_agent.agent.tools import (
    all_tool_schemas,
    configure_multi_agent,
    configure_policy,
    describe_policy,
)
from mini_agent.channels import validate_channels
from mini_agent.config import (
    DEFAULT_CONFIG_FILENAME,
    VALID_LOG_LEVELS,
    ConfigError,
    MiniConfig,
    MissingCredentialsError,
    load_config_report,
    write_config,
)
from mini_agent.memory import MemoryStore, configure_memory_store
from mini_agent.runtime.instance import Instance, InstanceLock, instance_path
from mini_agent.runtime.preflight import (
    BLOCKED,
    SETUP_REQUIRED,
    run_preflight,
    service_mode,
)
from mini_agent.sdk import MiniBot
from mini_agent.session.todo import TodoStore, configure_todo_store
from mini_agent.utils import credentials
from mini_agent.utils.errors import format_exception
from mini_agent.utils.netutil import is_loopback_url
from mini_agent.utils.observability import setup_logging

logger = logging.getLogger("mini_agent.cli")

# 界面退出后给优雅停机的上限；超过就强制结束，避免进程挂在后台。
_FORCE_EXIT_SECONDS = 15.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="mini_agent：面向生产的轻量异步 AI Agent 运行时",
        epilog=(
            "快速开始:\n"
            "  mini-agent              # 打开本机 Web 界面；首次运行进入设置页\n"
            "  mini-agent login        # 终端向导：把密钥存入系统钥匙串\n"
            "  mini-agent chat         # 终端对话\n"
            "  mini-agent doctor       # 环境诊断\n"
            "\n"
            "首次配置只需一次：Web 设置页或 login 会把密钥存进系统钥匙串，\n"
            "配置文件只保存与默认值不同的项；之后每次启动都不再询问。\n"
            "配置文件损坏时仍会启动修复页面，不会把用户挡在门外。\n"
            "\n"
            "等价的参数形式:\n"
            "  python -m mini_agent --web / --init / --doctor / -m \"你好\"\n"
            "\n"
            "环境变量（优先级高于配置文件）:\n"
            "  OPENAI_API_KEY   远程端点的密钥（覆盖钥匙串，适合 CI/容器）\n"
            "  OPENAI_BASE_URL  兼容端点，如 http://127.0.0.1:11434/v1\n"
            "  MODEL            模型名，如 gpt-4o-mini / qwen-plus\n"
            "  MINI_CONFIG      配置文件路径（等价于 -c）\n"
            "  MINI_SERVICE=1   无人值守模式：端口被占用直接报错，不自动避让\n"
            "  MINI_NO_BROWSER=1 启动时不自动打开浏览器\n"
            "  MINI_INSTANCE_FILE 单实例记录文件（默认 ~/.mini_agent/instances/）\n"
            "\n"
            "本地模型示例:\n"
            "  $env:OPENAI_BASE_URL=\"http://127.0.0.1:11434/v1\"; $env:MODEL=\"qwen2.5:7b\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("chat", "web", "login", "doctor", "init"),
        help="chat=终端对话；web=本机 Web；login=凭据向导；doctor=诊断；init=生成配置",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--init",
        action="store_true",
        help=f"生成配置模板（默认 {DEFAULT_CONFIG_FILENAME}，可用 -c 指定路径），不覆盖已有文件",
    )
    actions.add_argument(
        "--doctor",
        action="store_true",
        help="检查 Python、密钥、配置、目录、渠道与工具；不访问网络",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动 Web 前端（仅 web 渠道；默认 http://127.0.0.1:8767）",
    )
    parser.add_argument(
        "--web-port",
        "--port",
        dest="web_port",
        type=int,
        metavar="PORT",
        help="覆盖 Web 端口（显式指定时端口被占用直接报错，不自动避让）",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动 Web 时不自动打开浏览器",
    )
    parser.add_argument("-c", "--config", metavar="PATH", help="JSON 配置文件路径")
    parser.add_argument(
        "-m",
        "--message",
        metavar="TEXT",
        help="单轮模式：发送一条消息后退出；传 '-' 从 stdin 读取",
    )
    parser.add_argument(
        "-s",
        "--session",
        default="cli:local",
        metavar="KEY",
        help="会话键，交互/单轮模式均生效（默认 cli:local）",
    )
    parser.add_argument("--model", help="覆盖配置中的模型名")
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="覆盖配置中的兼容端点（如 http://127.0.0.1:11434/v1）",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖流式输出开关（默认取配置 stream=true）",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=sorted(VALID_LOG_LEVELS),
        help="覆盖日志级别（默认取配置中的 log_level）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验配置与渠道，打印生效配置和启用工具后退出",
    )
    parser.add_argument(
        "--version", action="version", version=f"mini-agent {__version__}"
    )
    return parser


def _resolve_command(args) -> str | None:
    """把子命令与参数归一成一个动作；冲突时打印一行错误并返回 None。"""
    command = args.command
    conflicts = []
    if args.message is not None and command in ("web", "start"):
        conflicts.append("子命令 web 不能与 -m/--message 同时使用")
    if args.init and command not in (None, "init"):
        conflicts.append("--init 不能与其他子命令同时使用")
    if args.doctor and command not in (None, "doctor"):
        conflicts.append("--doctor 不能与其他子命令同时使用")
    if args.web and command not in (None, "web"):
        conflicts.append("--web 不能与其他子命令同时使用")
    if args.web and args.message is not None:
        conflicts.append("--web 不能与 -m/--message 同时使用")
    if conflicts:
        for item in conflicts:
            print(f"配置错误: {item}", file=sys.stderr)
        return None

    if args.init or command == "init":
        return "init"
    if args.doctor or command == "doctor":
        return "doctor"
    if command == "login":
        return "login"
    if args.check:
        return "check"
    if args.message is not None:
        return "once"
    if args.web or command == "web":
        return "web"
    if command == "chat":
        return "chat"
    return "start"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = _resolve_command(args)
    if command is None:
        return 2

    if command == "init":
        return _run_init(args.config, model=args.model, base_url=args.base_url)

    web_mode = command in ("start", "web")
    instance_lock: InstanceLock | None = None
    if web_mode and not service_mode():
        # 复用检测必须先于配置加载：即使配置坏了，也不该再起第二个进程。
        instance_lock, running = InstanceLock.acquire(
            instance_path(_instance_key(args))
        )
        if running is not None:
            _reuse_instance(
                running,
                open_browser=not args.no_browser,
                requested_port=args.web_port,
            )
            return 0

    try:
        # 宽松加载：坏配置也要能把 Web 修复页面开起来；真正阻塞与否交给 preflight。
        report = load_config_report(args.config)
        overrides: dict[str, object] = {}
        if args.model:
            overrides["model"] = args.model
        if args.base_url:
            overrides["api_base"] = args.base_url
        if args.log_level:
            overrides["log_level"] = args.log_level
        if args.stream is not None:
            overrides["stream"] = args.stream
        if args.web_port is not None:
            overrides["web_port"] = args.web_port
        if web_mode:
            overrides["channels"] = ["web"]
        if overrides:
            report.config = replace(report.config, **overrides)
        config = report.config
    except ConfigError as exc:
        _release_instance(instance_lock)
        print(f"配置错误: {exc}", file=sys.stderr)
        print("提示: 运行 mini-agent doctor 查看完整诊断。", file=sys.stderr)
        return 2

    setup_logging(config.log_level)

    if command == "doctor":
        return _run_doctor(config, report)
    if command == "login":
        return _run_login(config, args.config)
    if command == "check":
        return _run_check(config)

    if command in ("once", "chat"):
        blocked_reason = _interactive_blocker(report)
        if blocked_reason:
            print(f"配置错误: {blocked_reason}", file=sys.stderr)
            return 2
        try:
            validate_channels(config.channels)
        except ValueError as exc:
            print(f"配置错误: {exc}", file=sys.stderr)
            return 2
        if command == "once":
            message = _read_message(args.message)
            if not message:
                print("配置错误: 消息为空", file=sys.stderr)
                return 2
            return _execute(_run_once(config, message, args.session))
        return _execute(_run_interactive(config, args.session))

    # 默认启动 / web：端口避让只用于本机自动模式。
    preflight = run_preflight(
        report,
        check_port=True,
        auto_port=args.web_port is None and not service_mode(),
    )
    if preflight.status == BLOCKED:
        _release_instance(instance_lock)
        print(f"启动失败: {preflight.error}", file=sys.stderr)
        return 2
    return _execute(
        _run_web(
            config,
            preflight,
            allow_setup=preflight.status == SETUP_REQUIRED,
            open_browser=not args.no_browser,
            config_path=args.config,
            lock=instance_lock,
        )
    )


def _instance_key(args) -> str:
    """单实例身份：显式配置按文件，否则按当前目录（工作区默认也是当前目录）。"""
    explicit = args.config or os.environ.get("MINI_CONFIG")
    return os.path.abspath(explicit) if explicit else os.getcwd()


def _release_instance(lock: InstanceLock | None) -> None:
    if lock is not None:
        lock.release()


def _reuse_instance(
    instance: Instance, *, open_browser: bool, requested_port: int | None = None
) -> None:
    """已在运行：只打开现有地址，不再启动第二个进程。"""
    base = instance.url.rstrip("/")
    print("=" * 64)
    print(f"mini_agent {instance.version or __version__} 已在运行")
    print(f"  地址:   {base}/")
    if instance.pid:
        print(f"  进程:   PID {instance.pid}")
    if requested_port and requested_port != instance.port:
        print(f"  端口:   --web-port {requested_port} 未生效，现有实例使用 {instance.port}")
    if not instance.ready:
        print("  状态:   待配置；设置链接只在首次启动的终端里打印")
    print("  说明:   已复用现有进程，避免两个进程同时写会话与记忆")
    print("  退出:   在 Web 界面点「退出」，或回到原终端按 Ctrl+C")
    print("=" * 64)
    sys.stdout.flush()
    if open_browser and not _no_browser():
        try:
            webbrowser.open(base + "/", new=2)
        except Exception:  # noqa: BLE001 - 无桌面环境时静默降级
            logger.debug("无法自动打开浏览器: %s", base)


def _interactive_blocker(report) -> str | None:
    """chat/单轮模式的启动门槛：坏配置或缺凭据都只给一行可操作提示。"""
    if report.needs_repair:
        return (
            "；".join(report.errors)
            + "。运行 mini-agent doctor 排查，或直接运行 mini-agent 打开修复页面"
        )
    try:
        report.config.validate_credentials()
    except MissingCredentialsError as exc:
        return f"{exc}（运行 mini-agent 可打开 Web 设置页）"
    return None


def _execute(coro) -> int:
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        return 0
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        _report_failure(exc)
        return 1
    return 0


def _run_init(
    config_path: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    """生成配置文件；绝不覆盖已有文件，避免误删用户配置。

    只写入与默认值不同的键，避免一个"全默认值"的项目配置把用户配置整体压掉。
    ``--model`` / ``--base-url`` 会写入模板，之后启动不必再重复输入端点。
    密钥走系统钥匙串，配置文件里永远不落密钥。
    """
    target = config_path or os.environ.get("MINI_CONFIG") or DEFAULT_CONFIG_FILENAME
    if os.path.exists(target):
        print(f"配置错误: 目标已存在，未覆盖: {os.path.abspath(target)}", file=sys.stderr)
        print(
            "如需重新生成，请先备份并删除该文件，或用 -c 指定另一个路径。",
            file=sys.stderr,
        )
        return 2

    try:
        overrides: dict[str, object] = {}
        if model:
            overrides["model"] = model
        if base_url:
            overrides["api_base"] = base_url
        template = replace(MiniConfig(), **overrides)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    try:
        target = write_config(template, target)
    except OSError as exc:
        print(f"配置错误: 无法写入 {target}: {exc}", file=sys.stderr)
        return 2

    print(f"已生成配置: {os.path.abspath(target)}")
    if model or base_url:
        print(
            "已写入: "
            + "，".join(
                part
                for part in (
                    f"model={template.model}" if model else "",
                    f"api_base={template.api_base}" if base_url else "",
                )
                if part
            )
        )
        print("之后启动会自动读取，不需要再重复输入模型和端点。")
    print("配置文件只包含与默认值不同的项；密钥请运行 mini-agent login 存入系统钥匙串。")
    print("完整选项说明见 config.example.json。")
    print()
    _print_first_run_guide()
    return 0


def _print_first_run_guide(stream=None) -> None:
    """首次使用指引：一次配置，之后每次启动都不再询问。"""
    say = lambda *parts: print(*parts, file=stream)  # noqa: E731 - 局部输出适配

    say("首次使用：只需要做一次配置，之后每次启动都直接进入对话")
    say()
    say("方式一（推荐）：本机 Web 设置页")
    say("   mini-agent")
    say("   启动后会打印一个带一次性令牌的设置链接，并在浏览器里打开；")
    say("   选择服务商、填模型与密钥、测试连接后保存即可。")
    say()
    say("方式二：终端向导（适合服务器或不便开浏览器时）")
    say("   mini-agent login")
    say("   密钥存入系统钥匙串，不会写进配置文件，也不会出现在日志里。")
    say()
    say("本地模型无需密钥：在 login 里选 Ollama，或直接在设置页填端点。")
    say("诊断：mini-agent doctor    配置与工具清单：python -m mini_agent --check")
    say(
        f"已经生成过 {DEFAULT_CONFIG_FILENAME} 时不要重复 --init；"
        "直接编辑该文件，或用 --model/--base-url 临时覆盖。"
    )


def _run_doctor(config: MiniConfig, config_report=None) -> int:
    """不建客户端、不发网络请求，逐项检查能否正常启动。"""
    issues = 0
    warnings = 0

    def report(level: str, label: str, detail: str) -> None:
        nonlocal issues, warnings
        if level == "错误":
            issues += 1
        elif level == "警告":
            warnings += 1
        print(f"[{level}] {label}: {detail}")

    print("=" * 64)
    print(f"mini_agent {__version__} 环境诊断")
    print("=" * 64)

    version = sys.version_info
    python_ok = version >= (3, 11)
    report(
        "正常" if python_ok else "错误",
        "Python",
        f"{version.major}.{version.minor}.{version.micro}"
        + ("" if python_ok else "（需要 3.11 或更高）"),
    )

    paths = list(config_report.paths) if config_report is not None else []
    if paths:
        report("正常", "配置文件", " -> ".join(os.path.abspath(item) for item in paths))
    else:
        report("提示", "配置文件", "未使用（全部取默认值与环境变量）")
    if config_report is not None and config_report.needs_repair:
        for issue in config_report.issues:
            if issue.severity == "error":
                report("错误", "配置解析", issue.message)

    if config.api_key:
        source_map = {
            "env": "环境变量 OPENAI_API_KEY",
            "keyring": "系统钥匙串",
            "file": "显式凭据文件",
            "config": "配置文件（建议迁移到钥匙串）",
        }
        source = source_map.get(config.credential_source, "未知来源")
        report("正常", "API 密钥", f"已配置（来自{source}，值不显示）")
    elif config.allow_no_api_key:
        report(
            "警告",
            "API 密钥",
            "未配置；allow_no_api_key=true 已豁免，但服务端仍可能拒绝请求",
        )
    elif is_loopback_url(config.api_base):
        report("正常", "API 密钥", "未配置；本地端点无需密钥")
    else:
        report(
            "错误",
            "API 密钥",
            "未配置；请设置 OPENAI_API_KEY，或改用本地端点 / allow_no_api_key",
        )

    report("正常", "模型", config.model)
    report("正常", "端点", config.api_base)

    try:
        validate_channels(config.channels)
        channels_detail = ", ".join(config.channels) or "(无)"
        if "echo" in config.channels:
            channels_detail += f"（echo 端口 {config.echo_port}）"
        if "web" in config.channels:
            channels_detail += f"（web 端口 {config.web_port}，仅本机）"
        report("正常", "渠道", channels_detail)
    except ValueError as exc:
        report("错误", "渠道", str(exc))

    ok, detail = _check_writable_dir(config.sessions_dir)
    report("正常" if ok else "错误", "会话目录", detail)

    if config.memory_enabled:
        if config.memory_backend == "mem0":
            if importlib.util.find_spec("mem0") is None:
                report(
                    "错误",
                    "长期记忆",
                    "memory_backend=mem0，但未安装依赖：pip install 'mem0ai>=2.2,<3'",
                )
            else:
                ok, detail = _check_writable_dir(config.mem0_dir)
                report(
                    "正常" if ok else "错误",
                    "长期记忆",
                    f"Mem0（{detail}；保存时调用模型抽取事实）",
                )
        else:
            ok, detail = _check_writable_dir(config.memory_dir)
            report("正常" if ok else "错误", "长期记忆", detail)
    else:
        report("提示", "长期记忆", "已禁用（memory_enabled=false）")

    if config.todos_enabled:
        ok, detail = _check_writable_dir(config.todos_dir)
        report("正常" if ok else "错误", "任务清单", detail)
    else:
        report("提示", "任务清单", "已禁用（todos_enabled=false）")

    report(
        "正常" if config.stream else "提示",
        "流式输出",
        "启用" if config.stream else "已禁用（stream=false）",
    )
    report(
        "正常",
        "超时与并发",
        f"整轮 {config.turn_timeout:g}s，单工具 {config.tool_timeout:g}s，"
        f"并发回合 {'不限' if config.max_concurrent_turns == 0 else config.max_concurrent_turns}",
    )

    if config.langfuse_enabled:
        if importlib.util.find_spec("langfuse") is None:
            report(
                "错误",
                "Langfuse",
                "已启用，但未安装依赖：pip install 'langfuse>=5,<6'",
            )
        else:
            keys = [
                name
                for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
                if os.environ.get(name)
            ]
            if len(keys) < 2:
                report(
                    "错误",
                    "Langfuse",
                    "已启用，但缺少环境变量 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY",
                )
            else:
                report(
                    "正常",
                    "Langfuse",
                    f"已启用（{config.langfuse_base_url or '默认云端地址'}，"
                    f"采样率 {config.langfuse_sample_rate:g}）",
                )
    else:
        report("提示", "Langfuse", "已禁用（langfuse_enabled=false）")

    if config.multi_agent_enabled:
        report(
            "正常",
            "多智能体",
            f"启用（并发 {config.max_concurrent_subagents}，"
            f"超时 {config.subagent_timeout:g}s，"
            f"结果上限 {config.subagent_max_result_chars} 字符）",
        )
    else:
        report("提示", "多智能体", "已禁用（multi_agent_enabled=false）")

    workspace = os.path.abspath(config.workspace)
    if os.path.isdir(workspace):
        report("正常", "工作区", workspace)
    else:
        report("错误", "工作区", f"{workspace} 不存在或不是目录")

    try:
        configure_policy(
            workspace=config.workspace,
            allow_shell=config.allow_shell,
            allow_private_urls=config.allow_private_urls,
            tool_approval=config.tool_approval,
        )
        # 诊断只枚举工具 schema，不实例化 Mem0/Qdrant；真实后端由 MiniBot 装配。
        configure_memory_store(
            MemoryStore(
                config.memory_dir,
                max_entries=config.memory_max_entries,
                enabled=config.memory_enabled,
            )
        )
        configure_todo_store(
            TodoStore(
                config.todos_dir,
                max_items=config.todo_max_items,
                enabled=config.todos_enabled,
            )
        )
        configure_multi_agent(config.multi_agent_enabled)
        policy = describe_policy()
        report(
            "正常",
            "工具策略",
            f"shell={'启用' if policy['allow_shell'] else '禁用'}；"
            f"内网 URL={'允许' if policy['allow_private_urls'] else '拒绝'}",
        )
        tools = [schema["function"]["name"] for schema in all_tool_schemas()]
        report("正常", "已启用工具", ", ".join(tools) or "(无)")
    except Exception as exc:  # noqa: BLE001 - 诊断命令要报告而不是抛出
        report("错误", "工具策略", str(exc))

    print("-" * 64)
    if issues:
        suffix = f"，{warnings} 个警告" if warnings else ""
        print(f"发现 {issues} 个必须解决的问题{suffix}。")
    else:
        suffix = f"（{warnings} 个警告）" if warnings else ""
        print(f"诊断通过，可以运行 python -m mini_agent{suffix}")
    print("=" * 64)

    missing_credentials = not (
        config.api_key or config.allow_no_api_key or is_loopback_url(config.api_base)
    )
    if missing_credentials:
        print()
        _print_first_run_guide()
    return 2 if issues else 0


def _check_writable_dir(path: str) -> tuple[bool, str]:
    """检查目录（或其最近的已存在父目录）是否可写；不实际创建目录。"""
    target = os.path.abspath(path)
    probe = target
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False, f"{target} 无法定位到可写目录"
        probe = parent
    if not os.path.isdir(probe):
        return False, f"{probe} 不是目录"
    if not os.access(probe, os.W_OK):
        return False, f"{probe} 不可写"
    if os.path.exists(target):
        return True, target
    return True, f"{target}（首次运行时自动创建）"


def _credential_status(config) -> str:
    if config.api_key:
        return "已配置"
    if config.allow_no_api_key:
        return "未配置（allow_no_api_key=true 已豁免）"
    if is_loopback_url(config.api_base):
        return "本地端点，无需密钥"
    return "未配置"


def _run_check(config) -> int:
    """不创建网络客户端；校验渠道名并输出脱敏后的有效配置。"""
    configure_policy(
        workspace=config.workspace,
        allow_shell=config.allow_shell,
        allow_private_urls=config.allow_private_urls,
        tool_approval=config.tool_approval,
    )
    # 只用于输出工具清单，不初始化 Mem0/Qdrant。
    configure_memory_store(
        MemoryStore(
            config.memory_dir,
            max_entries=config.memory_max_entries,
            enabled=config.memory_enabled,
        )
    )
    configure_todo_store(
        TodoStore(
            config.todos_dir,
            max_items=config.todo_max_items,
            enabled=config.todos_enabled,
        )
    )
    configure_multi_agent(config.multi_agent_enabled)

    payload = {
        "version": __version__,
        "config": config.describe(),
        "policy": describe_policy(),
        "enabled_tools": [
            schema["function"]["name"] for schema in all_tool_schemas()
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


async def _run_once(config, message: str, session_key: str) -> None:
    """单轮模式：不启动渠道，只走 SDK 路径。"""
    once_config = replace(config, channels=[])
    bot = MiniBot(once_config)
    try:
        if sys.stderr.isatty():
            # 提示走 stderr，保持 stdout 只有答案，方便管道消费
            print("（正在请求模型…）", file=sys.stderr, flush=True)
        streamed = False

        async def on_delta(delta: str) -> None:
            nonlocal streamed
            streamed = True
            print(delta, end="", flush=True)

        answer = await bot.chat(
            message,
            session_key=session_key,
            on_delta=on_delta if once_config.stream else None,
        )
        if streamed:
            print()
        else:
            print(answer)
    finally:
        await bot.aclose()


_LOGIN_PROVIDERS = (
    (
        "阿里云百炼 / DashScope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
    ),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("本地 Ollama", "http://127.0.0.1:11434/v1", "qwen2.5:7b"),
)


def _run_login(config: MiniConfig, config_path: str | None) -> int:
    """终端凭据向导：密钥只进系统钥匙串，配置只写差异。"""
    if not sys.stdin.isatty():
        print(
            "配置错误: login 需要交互终端。可改用 Web 设置页（直接运行 mini-agent），"
            "或在 CI/容器里设置 OPENAI_API_KEY 环境变量。",
            file=sys.stderr,
        )
        return 2

    print("=" * 64)
    print("mini_agent 凭据配置")
    print("  密钥只写入系统钥匙串；配置文件只保存与默认值不同的项。")
    print("=" * 64)
    for index, (label, base, _model) in enumerate(_LOGIN_PROVIDERS, 1):
        print(f"  {index}) {label:<22} {base}")
    print("  4) 自定义端点")
    choice = (input("选择服务商 [1]: ").strip() or "1")

    if choice in ("1", "2", "3"):
        _label, base_default, model_default = _LOGIN_PROVIDERS[int(choice) - 1]
    elif choice == "4":
        base_default, model_default = config.api_base, config.model
    else:
        print("配置错误: 无效的选择", file=sys.stderr)
        return 2

    model = (input(f"模型 [{model_default}]: ").strip() or model_default)
    api_base = (input(f"端点 [{base_default}]: ").strip() or base_default)

    hint = "可留空，本地服务通常不校验" if is_loopback_url(api_base) else "输入不回显"
    api_key = getpass.getpass(f"API 密钥（{hint}）: ").strip()

    existing, existing_source = credentials.load_api_key(api_base)
    if not api_key and not existing and not is_loopback_url(api_base):
        print("配置错误: 该端点需要 API 密钥。", file=sys.stderr)
        return 2

    try:
        candidate = replace(config, model=model, api_base=api_base, api_key=None)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    source = existing_source
    if api_key:
        try:
            source = credentials.save_api_key(api_base, api_key)
        except credentials.CredentialsError as exc:
            print(f"配置错误: {exc}", file=sys.stderr)
            return 2
        candidate.api_key = api_key
    else:
        candidate.api_key = existing
    candidate.credential_source = source

    if candidate.needs_setup():
        print("配置错误: 仍未拿到可用的 API 密钥。", file=sys.stderr)
        return 2

    try:
        target = write_config(candidate, config_path)
    except OSError as exc:
        print(f"配置错误: 无法写入配置文件: {exc}", file=sys.stderr)
        return 2

    saved_to = {
        "keyring": "系统钥匙串",
        "file": "显式凭据文件",
        "env": "环境变量",
    }.get(source, "本机凭据")
    print(f"密钥已保存到{saved_to}。")
    print(f"配置已写入: {os.path.abspath(target)}")

    if (input("现在测试连接？[Y/n]: ").strip().lower() not in ("n", "no")):
        result = asyncio.run(
            MiniBot.test_setup(
                model=model, api_base=api_base, api_key=api_key or None
            )
        )
        if result.get("ok"):
            print(f"连接成功：{result.get('model', model)}")
        else:
            print(f"连接失败：{result.get('error', '未知错误')}")
            print("配置已保存；可稍后运行 mini-agent 在设置页里修改。")

    print("完成。以后直接运行 mini-agent，不再询问。")
    return 0


async def _run_web(
    config,
    preflight=None,
    *,
    allow_setup: bool = False,
    open_browser: bool = True,
    config_path: str | None = None,
    lock: InstanceLock | None = None,
) -> None:
    """Web 模式：只装配 web 渠道，常驻服务直到 Ctrl+C 或界面退出。

    ``allow_setup=True`` 时即使缺凭据也照常启动：首页会 302 到设置页，
    ``/readyz`` 保持 503，配置保存后热重建 provider 并自动变为就绪。
    """
    bot: MiniBot | None = None
    ready = bool(preflight.ready) if preflight is not None else True
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    try:
        bot = MiniBot(
            config,
            allow_setup=allow_setup,
            preflight=preflight,
            config_path=config_path,
        )
        try:
            await bot.start()
        except OSError as exc:
            raise ConfigError(
                f"无法监听 127.0.0.1:{config.web_port}：{exc}；"
                "请换一个端口（--web-port 或配置 web_port）"
            ) from exc

        web = bot.channels.get("web")
        port = getattr(web, "bound_port", config.web_port)
        token = getattr(web, "setup_token", "")
        base_url = f"http://127.0.0.1:{port}"
        setup_url = f"{base_url}/setup#token={token}" if token and not ready else f"{base_url}/setup"

        # 监听成功后才登记端口；探活失败的记录会被下一次启动判为陈旧。
        if lock is not None:
            lock.write(
                port=port,
                url=base_url,
                version=__version__,
                ready=ready,
            )
        set_quit = getattr(web, "set_quit_handler", None)
        if callable(set_quit):
            loop = asyncio.get_running_loop()

            def quit_from_ui() -> None:
                logger.info("收到界面退出请求，开始优雅停机")
                _schedule_force_exit()
                loop.call_soon_threadsafe(stop.set)

            set_quit(quit_from_ui)

        print("=" * 64)
        print(f"mini_agent {__version__} 本机 Web")
        print(f"  地址:   {base_url}/")
        if not ready:
            print(f"  设置:   {setup_url}")
            print("  状态:   待配置（完成一次配置后即可直接进入对话）")
        else:
            print("  状态:   已就绪")
        if preflight is not None and preflight.port_shifted:
            print(f"  端口:   原端口被占用，已自动改用 {port}")
        print(f"  模型:   {config.model}")
        print("  探针:   /healthz（存活）  /readyz（就绪）")
        print("  安全:   仅监听本机回环；设置接口使用一次性令牌，保存后立即失效")
        print("  退出:   Ctrl+C 或界面右上角「退出」")
        print("=" * 64)
        # 重定向到文件/管道时 stdout 是块缓冲；设置链接必须立刻可见。
        sys.stdout.flush()

        if open_browser and not service_mode() and not _no_browser():
            target = base_url + "/" if ready else setup_url
            _open_browser_later(target)
        await stop.wait()
    finally:
        if bot is not None:
            await bot.aclose()
        _release_instance(lock)


def _schedule_force_exit() -> None:
    """界面退出后的兜底：优雅停机卡住时也要让进程真正退出。"""

    def force() -> None:
        logger.warning("优雅停机超过 %.0f 秒，强制退出进程", _FORCE_EXIT_SECONDS)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    timer = threading.Timer(_FORCE_EXIT_SECONDS, force)
    timer.daemon = True
    timer.start()


def _no_browser() -> bool:
    return os.environ.get("MINI_NO_BROWSER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _open_browser_later(url: str) -> None:
    """延迟打开浏览器：等服务完成监听，且失败不影响服务本身。"""

    def open_now() -> None:
        try:
            webbrowser.open(url, new=2)
        except Exception:  # noqa: BLE001 - 无桌面环境时静默降级
            logger.debug("无法自动打开浏览器: %s", url)

    timer = threading.Timer(1.0, open_now)
    timer.daemon = True
    timer.start()


async def _run_interactive(config, session_key: str) -> None:
    bot = MiniBot(config)
    cli_channel = bot.channels.get("cli")
    if cli_channel is not None:
        cli_channel.session_key = session_key
    policy = describe_policy()
    tool_names = [schema["function"]["name"] for schema in all_tool_schemas()]
    channels_text = ", ".join(bot.channels) if bot.channels else "(无)"

    print("=" * 64)
    print(f"mini_agent {__version__}（输入 /help 查看命令，/exit 退出）")
    print(f"  模型:     {config.model}")
    print(f"  端点:     {config.api_base}")
    print(f"  密钥:     {_credential_status(config)}")
    print(f"  渠道:     {channels_text}")
    if "web" in bot.channels:
        print(f"  Web:      http://127.0.0.1:{config.web_port}（仅本机）")
    print(f"  工作区:   {policy['workspace']}")
    print(f"  工具:     {', '.join(tool_names)}")
    print(f"  shell:    {'启用' if policy['allow_shell'] else '禁用'}")
    print(f"  会话目录: {config.sessions_dir}")
    if config.memory_enabled:
        memory_text = (
            f"启用（Mem0 -> {config.mem0_dir}）"
            if config.memory_backend == "mem0"
            else f"启用 {config.memory_dir}"
        )
    else:
        memory_text = "禁用"
    print(f"  长期记忆: {memory_text}")
    print(
        "  任务清单: "
        + (
            f"启用 {config.todos_dir}（最多 {config.todo_max_items} 项）"
            if config.todos_enabled
            else "禁用"
        )
    )
    print(f"  流式输出: {'启用' if config.stream else '禁用'}")
    print(
        f"  超时:     整轮 {config.turn_timeout:g}s / 单工具 {config.tool_timeout:g}s"
        f"（0 表示不限）"
    )
    print(
        "  Langfuse: "
        + (
            f"启用（{config.langfuse_base_url or '默认云端地址'}，"
            f"采样率 {config.langfuse_sample_rate:g}）"
            if config.langfuse_enabled
            else "禁用"
        )
    )
    print(
        "  多智能体: "
        + (
            f"启用（并发 {config.max_concurrent_subagents}，"
            f"超时 {config.subagent_timeout:g}s）"
            if config.multi_agent_enabled
            else "禁用"
        )
    )
    print(f"  会话键:   {session_key}")
    print("=" * 64)

    stop = asyncio.Event()
    _install_signal_handlers(stop)
    forward_task: asyncio.Task[None] | None = None
    try:
        await bot.start()
        cli = bot.channels.get("cli")
        closed = getattr(cli, "closed", None) if cli is not None else None
        if closed is not None:
            forward_task = asyncio.create_task(_forward_event(closed, stop))
        else:
            print("（当前没有 CLI 渠道，按 Ctrl+C 退出）")
        await stop.wait()
    finally:
        if forward_task is not None:
            forward_task.cancel()
            await asyncio.gather(forward_task, return_exceptions=True)
        await bot.aclose()
    print("已退出")


def _install_signal_handlers(stop: asyncio.Event) -> None:
    """SIGTERM/SIGINT 触发优雅停机；Windows 不支持时回退到 KeyboardInterrupt。"""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            continue


async def _forward_event(source: asyncio.Event, target: asyncio.Event) -> None:
    await source.wait()
    target.set()


def _read_message(value: str) -> str:
    if value != "-":
        return value.strip()
    raw = sys.stdin.buffer.read()
    if raw.startswith(b"\xff\xfe"):
        text = raw[2:].decode("utf-16-le", errors="replace")
    elif raw.startswith(b"\xfe\xff"):
        text = raw[2:].decode("utf-16-be", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.strip()


def _report_failure(exc: BaseException) -> None:
    logger.debug("运行失败详情", exc_info=exc)
    print(f"运行失败：{format_exception(exc)}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
