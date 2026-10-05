"""随机种子：三个随机源都要播种，且 `Generator` 不污染全局 RNG。

v1 只播了 torch（`TRAIN:428`、`SFT:382,488`），DataLoader / Python `random` /
NumPy 仍是未播种的随机源，这就是"同种子两次结果不同"的来源。
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from my_llm.utils.seed import get_generator, seed_worker, set_seed


def _draw_all() -> tuple[float, float, float]:
    """从三个随机源各取一个数。

    Returns:
        `(python_random, numpy_random, torch_random)`。
    """
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )


def test_set_seed_makes_three_sources_reproducible() -> None:
    """同一种子重放，`random` / `numpy` / `torch` 三条序列都必须逐位一致。"""
    set_seed(0)
    first = _draw_all()
    set_seed(0)
    second = _draw_all()
    msg = f"同一种子两次结果不一致: {first} vs {second}"
    assert first == second, msg


def test_set_seed_changes_the_sequence() -> None:
    """不同种子必须产出不同序列（否则说明种子根本没生效）。"""
    set_seed(0)
    first = _draw_all()
    set_seed(1)
    second = _draw_all()
    msg = f"不同种子结果相同，种子未生效: {first}"
    assert first != second, msg


def test_negative_seed_raises() -> None:
    """负种子直接报错，而不是被静默接受。"""
    with pytest.raises(ValueError, match="seed"):
        set_seed(-1)


def test_get_generator_leaves_global_rng_untouched() -> None:
    """用 `Generator` 采样不推进全局 RNG（v1 靠全局 `manual_seed`，采样会挪动它）。"""
    set_seed(0)
    baseline = torch.rand(4).clone()

    set_seed(0)
    generator = get_generator(7)
    torch.rand(3, generator=generator)
    after = torch.rand(4)

    msg = f"用 Generator 采样后全局 RNG 被挪动了: {baseline.tolist()} vs {after.tolist()}"
    assert torch.equal(baseline, after), msg


def test_get_generator_is_seed_deterministic() -> None:
    """同种子构造的两个 `Generator` 产出相同序列。"""
    first = torch.rand(5, generator=get_generator(7))
    second = torch.rand(5, generator=get_generator(7))
    msg = "同种子的 Generator 应产出相同序列"
    assert torch.equal(first, second), msg


def test_seed_worker_is_reproducible_and_worker_dependent() -> None:
    """`worker_init_fn`：同 worker 编号可复现，不同编号种子不同。"""
    torch.manual_seed(5)
    seed_worker(0)
    first = torch.rand(3).clone()

    torch.manual_seed(5)
    seed_worker(0)
    second = torch.rand(3).clone()
    msg = "同一 worker 编号应可复现"
    assert torch.equal(first, second), msg

    torch.manual_seed(5)
    seed_worker(1)
    other = torch.rand(3)
    msg = "不同 worker 编号应拿到不同种子"
    assert not torch.equal(first, other), msg
