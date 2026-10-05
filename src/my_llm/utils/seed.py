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

import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """播种全部随机源。

    Args:
        seed: 随机种子（非负整数）。
        deterministic: 是否额外启用 torch 确定性算法与关闭 cuDNN benchmark。

    Raises:
        ValueError: `seed` 为负数。
    """
    if seed < 0:
        msg = f"seed 必须是非负整数，收到 {seed}"
        raise ValueError(msg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 只在这里打开：确定性算法对某些算子没有实现，会在**前向时**抛错，
        # 因此不能作为默认行为（默认播种就能复现绝大多数实验）。
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int, *, device: torch.device | str = "cpu") -> torch.Generator:
    """构造与本轮实验绑定的 Generator，避免采样污染全局 RNG。

    Args:
        seed: 随机种子。
        device: Generator 所在设备。

    Returns:
        torch.Generator 实例。
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 `worker_init_fn`，保证多进程加载也可复现。

    Args:
        worker_id: 子进程编号。

    Returns:
        None；副作用是就地播种该 worker 的随机源。

    Raises:
        RuntimeError: 依赖 numpy/torch 但二者不可用时。
    """
    # torch 已经为每个 worker 派发了不同种子；这里把它同步给另外两个随机源，
    # 否则 shuffle / 采样在 num_workers > 0 时不可复现。
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


__all__ = ["get_generator", "seed_worker", "set_seed"]
