"""凭据读取与保存：系统钥匙串优先，密钥永远不写进配置文件。

保存优先级：
    1. OS Keyring（Windows 凭据管理器 / macOS Keychain / Secret Service）
    2. 环境变量（只读，用于 CI、容器与临时覆盖）
    3. 显式凭据文件（仅在设置 MINI_CREDENTIALS_FILE 时启用，权限 0600）

读取时环境变量优先：CI 和容器里 ``OPENAI_API_KEY`` 应当能覆盖钥匙串里的旧值，
否则轮换密钥会变得不可预测。钥匙串按 ``api_base`` 的主机名分开保存，
切换服务商不会误用上一家的密钥。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

ENV_API_KEY = "OPENAI_API_KEY"
ENV_CREDENTIALS_FILE = "MINI_CREDENTIALS_FILE"
KEYRING_SERVICE = "mini-agent"

_KEYRING_UNAVAILABLE = (
    "系统钥匙串不可用（keyring 未安装或后端初始化失败），无法安全保存凭据。"
    "请先执行 pip install keyring，并确认系统凭据服务可用；"
    f"或设置 {ENV_CREDENTIALS_FILE} 指向一个仅当前用户可读的文件。"
)


class CredentialsError(RuntimeError):
    """凭据无法读取或保存；消息里绝不包含密钥本身。"""


def _username(api_base: str) -> str:
    host = urlsplit(api_base).netloc or "default"
    return host.lower()


def _keyring():
    """延迟导入：未安装 keyring 时不应影响普通启动。"""
    try:
        import keyring  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - 缺少依赖或后端初始化失败都视为不可用
        return None
    return keyring


def keyring_available() -> bool:
    keyring = _keyring()
    if keyring is None:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:  # noqa: BLE001
        return False
    name = type(backend).__name__.lower()
    if "fail" in name or "null" in name:
        return False
    # keyring 的 fail/null 兜底后端 priority 为 0；真实后端为正数。
    # 只看类名不够：fail backend 的类名就叫 Keyring。
    try:
        return float(getattr(backend, "priority", 1)) > 0
    except (TypeError, ValueError):
        return True


def _read_file(path: str) -> dict[str, str]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("无法读取凭据文件 %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value}


def _write_file(path: str, username: str, key: str) -> None:
    data = _read_file(path)
    data[username] = key
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True, mode=0o700)
    fd, temp_path = tempfile.mkstemp(prefix=".credentials-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        if os.name == "nt":
            _restrict_windows_acl(temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _restrict_windows_acl(path: str) -> None:
    """把文件 ACL 收紧到当前用户；失败只告警，不阻断保存。"""
    try:
        import subprocess

        user = os.environ.get("USERNAME")
        if not user:
            return
        subprocess.run(
            ["icacls", path, "/inheritance:r", "/grant:r", f"{user}:F"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("收紧凭据文件权限失败: %s", exc)


def load_api_key(api_base: str) -> tuple[str | None, str]:
    """返回 ``(key, source)``；source 为 env / keyring / file / 空字符串。"""
    env_value = os.environ.get(ENV_API_KEY)
    if env_value and env_value.strip():
        return env_value.strip(), "env"

    keyring = _keyring()
    if keyring is not None:
        try:
            stored = keyring.get_password(KEYRING_SERVICE, _username(api_base))
        except Exception as exc:  # noqa: BLE001 - 钥匙串后端可能临时不可用
            logger.warning("读取系统钥匙串失败: %s", exc)
            stored = None
        if stored and stored.strip():
            return stored.strip(), "keyring"

    file_path = os.environ.get(ENV_CREDENTIALS_FILE)
    if file_path:
        stored = _read_file(file_path).get(_username(api_base))
        if stored and stored.strip():
            return stored.strip(), "file"
    return None, ""


def save_api_key(api_base: str, key: str) -> str:
    """保存密钥，返回实际存储位置（keyring / file）。"""
    key = (key or "").strip()
    if not key:
        raise CredentialsError("密钥不能为空")
    username = _username(api_base)

    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.set_password(KEYRING_SERVICE, username, key)
            return "keyring"
        except Exception as exc:  # noqa: BLE001 - 后端不可用时再尝试文件
            logger.warning("写入系统钥匙串失败: %s", exc)

    file_path = os.environ.get(ENV_CREDENTIALS_FILE)
    if file_path:
        _write_file(file_path, username, key)
        return "file"
    raise CredentialsError(_KEYRING_UNAVAILABLE)


