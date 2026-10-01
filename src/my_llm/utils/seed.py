"""可复现性工具（v2 新增）。

v1 只有零散的 `torch.manual_seed(123)`（`TRAIN:428`、`SFT:382,488`、`LOAD:481,506`），问题有三：
1. 没有播种 Python `random` 与 NumPy，DataLoader / 采样路径仍有随机源；
2. 没有种子 CLI 参数，改种子要改源码（同可以的硬编码问题）；
3. 没有 `torch.use_deterministic_algorithms` / cuDNN 开关，严格复现不可控。

这里统一到一个入口：`set_seed(seed, deterministic=True)`。
`deterministic=True` 会打开确定性算法，**可能报错**（某些算子无确定性实现），
因此作为可选参数而不是默认行为。
"""

from __future__ import annotations

import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """播种全部随机源。

    Args:
        seed: 随机种子（非负整数）。
        deterministic: 是否额外启用 torch 确定性算法与关闭 cuDNN benchmark。

    Raises:
        ValueError: `seed` 为负数。
    """
    raise NotImplementedError


def get_generator(seed: int, *, device: torch.device | str = "cpu") -> torch.Generator:
    """构造与本轮实验绑定的 Generator，避免采样污染全局 RNG。

    Args:
        seed: 随机种子。
        device: Generator 所在设备。

    Returns:
        torch.Generator 实例。
    """
    raise NotImplementedError


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 `worker_init_fn`，保证多进程加载也可复现。

    Args:
        worker_id: 子进程编号。

    Returns:
        None；副作用是就地播种该 worker 的随机源。

    Raises:
        RuntimeError: 依赖 numpy/torch 但二者不可用时。
    """
    raise NotImplementedError


__all__ = ["get_generator", "seed_worker", "set_seed"]
