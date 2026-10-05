"""日志（v2 新增）。

v1 全程 `print()`（例如 `TRAIN:349`、`SFT:117-121`），没有级别、没有格式、无法重定向。
统一走 `logging`：库里只 `get_logger`，配置由入口脚本决定。
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

LOGGER_NAME = "my_llm"

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """取一个位于 `my_llm.` 命名空间下的 logger。

    Args:
        name: 子模块名，例如 `"train.trainer"`；`None` 时返回包根 logger。

    Returns:
        logging.Logger 实例。
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO, *, stream: TextIO | None = None) -> None:
    """配置根 logger 的输出级别与格式（入口脚本调用）。

    库里只 `get_logger`，不调用本函数 —— 是否输出、输出到哪里由入口脚本决定。

    Args:
        level: 日志级别。
        stream: 输出流，默认 `sys.stderr`。

    Returns:
        None。
    """
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))

    # 装着 LOGGER_NAME 这一"包根"配置；库里的 logger 都是它的子 logger，会自动继承。
    # 先清 handler 是为了让重复调用（脚本里调一次、测试里再调一次）不叠加重复输出。
    root = logging.getLogger(LOGGER_NAME)
    for old in root.handlers[:]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)


__all__ = ["LOGGER_NAME", "configure_logging", "get_logger"]
