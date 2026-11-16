"""启动预检：把"能不能用"归纳成 ready / setup_required / blocked 三种状态。

设计原则（对应产品要求）：

- **blocked**：环境问题，界面上做什么都没用。Python 过旧、核心依赖导入失败、
  状态目录不可写、端口被占用且不允许避让。必须给出原因和一条修复命令。
- **setup_required**：程序能跑，只是还没配置好。缺密钥/模型、配置文件损坏
  都归到这里，启动 Web 修复页面让用户就地解决，而不是把人挡在门外。
- **ready**：依赖、配置、凭据、端口都就绪，直接进入对话。

端口避让只用于本机自动模式；显式 ``--port``、Docker/systemd 等无人值守场景
直接报错，避免健康检查或反向代理连到"看起来起来了"的错误端口。
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import tempfile
from dataclasses import asdict, dataclass

from ..config import ConfigReport
from ..utils.netutil import is_loopback_url

READY = "ready"
SETUP_REQUIRED = "setup_required"
BLOCKED = "blocked"

_DEPENDENCY_REPAIR = (
    "python -m pip install --force-reinstall -r mini_agent/requirements.txt"
)
_CORE_MODULES = ("openai", "pydantic_core._pydantic_core")
_PORT_ATTEMPTS = 20


class PortUnavailableError(RuntimeError):
    """端口不可用。消息可直接展示给用户。"""


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # ok / warn / error
    detail: str
    action: str = ""


@dataclass
class PreflightResult:
    status: str
    checks: list[CheckResult]
    report: ConfigReport
    web_port: int = 0
    port_shifted: bool = False
    repair: bool = False
    setup_reason: str = ""

    @property
    def config(self):
        return self.report.config

    @property
    def ready(self) -> bool:
        return self.status == READY

    @property
    def reason(self) -> str:
        for check in self.checks:
            if check.status == "error":
                return check.detail
        return self.setup_reason

    @property
    def action(self) -> str:
        for check in self.checks:
            if check.status == "error":
                return check.action
        return ""

    @property
    def error(self) -> str:
        """blocked 时给 CLI 的一行提示（原因 + 修复动作）。"""
        reason = self.reason or "启动预检失败"
        return f"{reason}；修复：{self.action}" if self.action else reason

    def as_dict(self) -> dict:
        """可安全暴露给 Web 设置页/健康检查的摘要，不含任何密钥。"""
        config = self.report.config
        return {
            "status": self.status,
            "ready": self.ready,
            "repair": self.repair,
            "reason": self.reason,
            "action": self.action,
            "web_port": self.web_port,
            "port_shifted": self.port_shifted,
            "checks": [asdict(item) for item in self.checks],
            "config": {
                "model": config.model,
                "api_base": config.api_base,
                "credential_source": config.credential_source,
                "needs_setup": config.needs_setup(),
                "paths": list(self.report.paths),
            },
            "issues": [asdict(item) for item in self.report.issues],
        }


def service_mode() -> bool:
    """Docker/systemd/K8s 等无人值守环境：端口必须严格按配置来。"""
    if _truthy(os.environ.get("MINI_SERVICE")):
        return True
    if os.environ.get("INVOCATION_ID") or os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    if os.environ.get("NOTIFY_SOCKET"):
        return True
    return os.path.exists("/.dockerenv")


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in {"1", "true", "yes", "on"}


def port_is_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if os.name != "nt":
            # Windows 上 SO_REUSEADDR 会允许抢占已被监听的端口，反而让探测失真。
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(
    host: str,
    preferred: int,
    *,
    auto: bool = True,
    attempts: int = _PORT_ATTEMPTS,
) -> tuple[int, bool]:
    """返回 ``(端口, 是否发生避让)``；不允许避让时占用即抛错。"""
    if preferred == 0:
        return 0, False
    limit = max(1, attempts) if auto else 1
    tried: list[int] = []
    for offset in range(limit):
        port = preferred + offset
        if port > 65535:
            break
        if port_is_free(host, port):
            return port, offset > 0
        tried.append(port)
    detail = (
        f"Web 端口 {preferred} 已被占用，且当前模式不允许自动避让"
        if not auto
        else f"Web 端口 {preferred} 起连续 {len(tried)} 个端口都被占用"
    )
    raise PortUnavailableError(detail)


def run_preflight(
    report: ConfigReport,
    *,
    check_port: bool = False,
    auto_port: bool = True,
    host: str = "127.0.0.1",
) -> PreflightResult:
    """执行检查并返回三态结果；不会创建 MiniBot、不会访问网络。"""
    checks: list[CheckResult] = [_check_python(), _check_dependencies()]
    checks.append(_check_config(report))
    checks.append(_check_state_dirs(report.config))

    web_port = report.config.web_port
    port_shifted = False
    if check_port:
        port_check, web_port, port_shifted = _check_port(
            host, web_port, auto=auto_port
        )
        checks.append(port_check)

    checks.append(_check_credentials(report))

    blocking = any(item.status == "error" for item in checks)
    if blocking:
        status = BLOCKED
    elif report.needs_repair or report.config.needs_setup():
        status = SETUP_REQUIRED
    else:
        status = READY

    setup_reason = ""
    if status == SETUP_REQUIRED:
        setup_reason = (
            "配置文件无法解析，需要修复"
            if report.needs_repair
            else "尚未完成首次配置（缺少 API 密钥或模型端点）"
        )
    return PreflightResult(
        status=status,
        checks=checks,
        report=report,
        web_port=web_port,
        port_shifted=port_shifted,
        repair=report.needs_repair,
        setup_reason=setup_reason,
    )


def _check_python() -> CheckResult:
    version = sys.version_info
    label = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        return CheckResult(
            "python", "error", f"Python {label} 过旧，需要 3.11 或更高", "python --version"
        )
    return CheckResult("python", "ok", f"Python {label}")


def _check_dependencies() -> CheckResult:#
    broken: list[str] = []
    for name in _CORE_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - 二进制扩展损坏也在此暴露
            broken.append(f"{name}（{type(exc).__name__}: {exc}）")
    if broken:
        return CheckResult(
            "dependencies",
            "error",
            "核心依赖不可用: " + "；".join(broken),
            _DEPENDENCY_REPAIR,
        )
    return CheckResult("dependencies", "ok", "核心依赖可导入")


def _check_config(report: ConfigReport) -> CheckResult:#
    if report.needs_repair:
        return CheckResult(
            "config",
            "warn",
            "；".join(report.errors),
            "打开设置页修复配置，或备份后删除损坏文件重新生成",
        )
    if report.paths:
        return CheckResult("config", "ok", "配置来源: " + " -> ".join(report.paths))
    return CheckResult("config", "ok", "未使用配置文件，全部取默认值与环境变量")


def _check_state_dirs(config) -> CheckResult:
    targets = [("sessions_dir", config.sessions_dir)]
    if config.memory_enabled:
        targets.append(("memory_dir", config.memory_dir))
        if config.memory_backend == "mem0":
            targets.append(("mem0_dir", config.mem0_dir))
    if config.todos_enabled:
        targets.append(("todos_dir", config.todos_dir))
    targets.append(("knowledge_dir", config.knowledge_dir))
    for label, path in targets:
        try:
            os.makedirs(path, exist_ok=True)
            fd, probe = tempfile.mkstemp(prefix=".mini-agent-", dir=path)
            os.close(fd)
            os.unlink(probe)
        except OSError as exc:
            return CheckResult(
                "state_dir",
                "error",
                f"{label} 不可写: {path}（{exc.strerror or exc}）",
                f"检查目录权限或改用 -c 配置可写路径: {path}",
            )
    return CheckResult("state_dir", "ok", "会话/记忆/清单/知识库目录可写")


def _check_port(host: str, preferred: int, *, auto: bool) -> tuple[CheckResult, int, bool]:
    try:
        port, shifted = choose_port(host, preferred, auto=auto)
    except PortUnavailableError as exc:
        return (
            CheckResult(
                "port",
                "error",
                str(exc),
                "改用 --web-port <空闲端口>，或停止占用该端口的程序",
            ),
            preferred,
            False,
        )
    if shifted:
        return (
            CheckResult("port", "warn", f"端口 {preferred} 被占用，已改用 {port}"),
            port,
            True,
        )
    return CheckResult("port", "ok", f"端口 {port} 可用"), port, False


def _check_credentials(report: ConfigReport) -> CheckResult:
    config = report.config
    if config.api_key:
        source = config.credential_source or "配置"
        return CheckResult("credentials", "ok", f"API 密钥已配置（来源: {source}）")
    if config.allow_no_api_key:
        return CheckResult("credentials", "warn", "未配置密钥，allow_no_api_key=true")
    if is_loopback_url(config.api_base):
        return CheckResult("credentials", "ok", "本地端点无需密钥")
    return CheckResult(
        "credentials",
        "warn",
        "未配置 API 密钥",
        "运行 mini-agent login，或在设置页填写",
    )
