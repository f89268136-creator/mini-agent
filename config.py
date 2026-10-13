"""配置（对应 nanobot 的 config/schema.py + loader.py 的最小版）。

nanobot 生产版用 Pydantic 定义配置模型，从 ~/.nanobot/config.json 和环境
变量加载；本教学版用 dataclass + 手工解析，不引入任何三方依赖。

优先级（后者覆盖前者）：默认值 < JSON 文件 < 环境变量。
环境变量覆盖是部署时最常用的手段——不改任何文件就能切换模型/密钥。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Callable

# 从旧版 agent.py 原样搬来的默认人设（保证行为兼容）
DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的 AI 助手。需要实时信息或执行操作时，请使用提供的工具；"
    "拿到工具结果后，再用中文给出最终回答。"
)

# 未显式传 path 时，默认在当前工作目录找这个文件
DEFAULT_CONFIG_FILENAME = ".mini_agent.json"


@dataclass
class MiniConfig:
    """全部可配置项。字段名同时充当 JSON 文件里的键名。"""

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str = "https://api.openai.com/v1"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_iterations: int = 10  # 单轮对话里工具调用的最大循环次数
    history_limit: int = 50  # 发给模型的历史消息上限
    sessions_dir: str = ".mini_agent_sessions"  # 会话 JSONL 存放目录
    channels: list[str] = field(default_factory=lambda: ["cli"])
    echo_port: int = 8766  # echo 渠道监听的 TCP 端口
    retry_max_attempts: int = 3  # LLM 请求重试次数
    retry_base_delay: float = 0.5  # 重试的指数退避基数（秒）
    timeout: float = 60.0  # 单次 LLM 请求超时（秒）


def load_config(path: str | None = None) -> MiniConfig:
    """按"默认值 -> JSON -> 环境变量"的顺序装配配置。"""
    cfg = MiniConfig()

    # ① JSON 文件：显式 path 优先；否则 CWD 下存在默认文件就读它
    config_path = path or (
        DEFAULT_CONFIG_FILENAME if os.path.exists(DEFAULT_CONFIG_FILENAME) else None
    )
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            # 教学版对坏配置的态度：告警 + 回退默认，不崩溃
            print(f"[mini_agent] 配置文件解析失败（{config_path}）：{exc}，使用默认配置")
            data = {}
        valid_fields = {f.name for f in fields(MiniConfig)}
        for key, value in data.items():
            if key in valid_fields:
                setattr(cfg, key, value)
            else:
                print(f"[mini_agent] 未知配置项已忽略：{key}")

    # ② 环境变量：优先级最高，逐个覆盖同名字段
    _apply_env(cfg)
    return cfg


def _apply_env(cfg: MiniConfig) -> None:
    str_vars = {
        "MODEL": "model",
        "OPENAI_API_KEY": "api_key",
        "OPENAI_BASE_URL": "api_base",
        "MINI_SESSIONS_DIR": "sessions_dir",
    }
    for env_name, field_name in str_vars.items():
        value = os.environ.get(env_name)
        if value:
            setattr(cfg, field_name, value)

    # 渠道列表用逗号分隔："MINI_CHANNELS=cli,echo"
    raw_channels = os.environ.get("MINI_CHANNELS")
    if raw_channels:
        cfg.channels = [c.strip() for c in raw_channels.split(",") if c.strip()]

    int_vars = {
        "MAX_ITERATIONS": "max_iterations",
        "HISTORY_LIMIT": "history_limit",
        "ECHO_PORT": "echo_port",
        "MINI_RETRY_MAX_ATTEMPTS": "retry_max_attempts",
    }
    for env_name, field_name in int_vars.items():
        _set_number(cfg, env_name, field_name, os.environ.get(env_name), int)

    float_vars = {
        "MINI_RETRY_BASE_DELAY": "retry_base_delay",
        "MINI_TIMEOUT": "timeout",
    }
    for env_name, field_name in float_vars.items():
        _set_number(cfg, env_name, field_name, os.environ.get(env_name), float)


def _set_number(
    cfg: MiniConfig, env_name: str, field_name: str, raw: str | None, cast: Callable[[str], object]
) -> None:
    """解析数字型环境变量；格式不对时告警并保留原值。"""
    if not raw:
        return
    try:
        setattr(cfg, field_name, cast(raw))
    except ValueError:
        print(f"[mini_agent] 环境变量 {env_name} 不是合法数字，已忽略：{raw}")
