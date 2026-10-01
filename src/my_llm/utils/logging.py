"""日志（v2 新增）。

v1 全程 `print()`（例如 `TRAIN:349`、`SFT:117-121`），没有级别、没有格式、无法重定向。
统一走 `logging`：库里只 `get_logger`，配置由入口脚本决定。
"""

from __future__ import annotations

import logging
from typing import TextIO

LOGGER_NAME = "my_llm"


def get_logger(name: str | None = None) -> logging.Logger:
    """取一个位于 `my_llm.` 命名空间下的 logger。

    Args:
        name: 子模块名，例如 `"train.trainer"`；`None` 时返回包根 logger。

    Returns:
        logging.Logger 实例。
    """
    raise NotImplementedError


def configure_logging(level: int | str = logging.INFO, *, stream: TextIO | None = None) -> None:
    """配置根 logger 的输出级别与格式（入口脚本调用）。

    Args:
        level: 日志级别。
        stream: 输出流，默认 `sys.stderr`。

    Returns:
        None。
    """
    raise NotImplementedError


__all__ = ["LOGGER_NAME", "configure_logging", "get_logger"]
