"""本机单实例复用：同一份配置只保留一个 Web 进程。

再次启动时先探测已经运行的实例：探活成功就只打开它的地址并退出，
不再创建第二个进程。两份进程共享会话/记忆目录会互相覆盖，所以这里
宁可复用也不并行。记录文件按配置路径（未显式指定时按当前目录）分片，
不同项目可以各自运行自己的实例。

探测使用 ``/healthz``，不依赖 PID 存活判断（Windows 上 ``os.kill`` 的
语义不适合探活）；记录里带端口但探活失败即视为陈旧记录，直接接管。
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_PROBE_TIMEOUT = 1.0
_STARTUP_WAIT = 8.0
_FRESH_SECONDS = 15.0


@dataclass(frozen=True)
class Instance:
    """一个探活成功的运行实例；字段全部来自记录文件，不含密钥。"""

    pid: int
    port: int
    url: str
    version: str = ""
    ready: bool = True


def instance_path(key: str) -> Path:
    """记录文件路径；``MINI_INSTANCE_FILE`` 可覆盖，便于测试或并行调试。"""
    override = os.environ.get("MINI_INSTANCE_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    digest = hashlib.sha1(os.path.abspath(key).encode("utf-8")).hexdigest()[:12]
    return Path("~/.mini_agent/instances").expanduser() / f"{digest}.json"


def probe(port: int) -> dict | None:
    """确认端口上跑的是 mini_agent 而不是别的程序。"""
    if not port:
        return None
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=_PROBE_TIMEOUT
    )
    try:
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        if response.status != 200:
            return None
        payload = json.loads(response.read(4096).decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    if not isinstance(payload, dict) or payload.get("service") != "mini_agent":
        return None
    return payload


def read_record(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("service") != "mini_agent":
        return None
    return data


def _fresh(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime < _FRESH_SECONDS
    except OSError:
        return False


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


class InstanceLock:
    """记录文件的独占所有权；进程退出时只删除属于自己的记录。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._owned = False

    @classmethod
    def acquire(
        cls, path: Path, *, wait: float = _STARTUP_WAIT
    ) -> tuple["InstanceLock | None", Instance | None]:
        """返回 ``(lock, None)`` 表示拿到所有权，``(None, instance)`` 表示复用。

        两者都是 None 时表示记录文件不可写；此时退化为不守护，不能因为
        单实例机制本身让用户启动不了。
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None, None

        deadline = time.monotonic() + wait
        while True:
            try:
                handle = os.open(
                    path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                record = read_record(path)
                port = record.get("port") if record else None
                pid = int(record.get("pid") or 0) if record else 0
                if port and pid:
                    if probe(int(port)):
                        return None, Instance(
                            pid=pid,
                            port=int(port),
                            url=str(record.get("url") or f"http://127.0.0.1:{port}"),
                            version=str(record.get("version") or ""),
                            ready=bool(record.get("ready", True)),
                        )
                    # 有端口却探活失败：上一个进程已经退出或端口被别的程序占用。
                    _remove(path)
                    continue
                if _fresh(path) and time.monotonic() < deadline:
                    # 另一个进程正在启动，等它写完端口再做判断。
                    time.sleep(0.15)
                    continue
                _remove(path)
                continue
            except OSError:
                return None, None
            else:
                os.close(handle)
                lock = cls(path)
                lock._owned = True
                return lock, None

    def write(
        self,
        *,
        port: int,
        url: str,
        version: str = "",
        ready: bool = True,
    ) -> None:
        if not self._owned:
            return
        payload = {
            "service": "mini_agent",
            "pid": os.getpid(),
            "port": int(port),
            "url": url,
            "version": version,
            "ready": bool(ready),
        }
        try:
            temp_path = ""
            fd, temp_path = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".instance-", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except OSError:
            # 写记录失败只影响"下次能否复用"，不影响本次服务。
            if temp_path:
                _remove(Path(temp_path))

    def release(self) -> None:
        if not self._owned:
            return
        self._owned = False
        record = read_record(self.path)
        if record is not None and record.get("pid") != os.getpid():
            return
        _remove(self.path)
