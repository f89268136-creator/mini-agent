"""进程级日志配置。

库代码只通过 ``logging.getLogger(__name__)`` 记录事件，不直接向终端打印；
是否输出、输出到哪里由应用层决定。CLI 入口调用 :func:`setup_logging`，
SDK 使用者可以接入自己的日志系统，或者直接调用本函数获得一致的格式。
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "mini_agent"
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def setup_logging(level: str = "INFO", *, force: bool = False) -> None:
    """配置 ``mini_agent`` 命名空间的日志输出（stderr）。

    ``force=True`` 会移除已有 StreamHandler 后重新挂载，便于测试或宿主程序
    切换日志级别；默认情况下重复调用只更新级别，不重复添加 handler。
    """
    normalized = str(level).upper()
    if normalized not in _VALID_LEVELS:
        raise ValueError(f"非法日志级别: {level!r}（可选: {', '.join(sorted(_VALID_LEVELS))}）")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, normalized))
    logger.propagate = False

    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    if force:
        for handler in stream_handlers:
            logger.removeHandler(handler)
        stream_handlers = []

    if not stream_handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
