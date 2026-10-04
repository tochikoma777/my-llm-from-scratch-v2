"""warmup + cosine 学习率调度。

断言分三层：warmup 段线性上升、余弦段单调下降、末步不低于 `min_lr_ratio × base_lr`；
再加一条"LambdaLR 逐步取值 == `warmup_cosine_lr`"把两个入口对起来。
"""

from __future__ import annotations

import pytest
import torch

from my_llm.train import get_cosine_schedule_with_warmup, warmup_cosine_lr

BASE_LR = 0.1
WARMUP = 10
TOTAL = 100
MIN_RATIO = 0.1


def _schedule(min_lr_ratio: float = 0.0) -> list[float]:
    """整条 lr 曲线（按 `base_lr = 1.0` 归一化）。"""
    return [
        warmup_cosine_lr(s, 1.0, WARMUP, TOTAL, min_lr_ratio=min_lr_ratio) for s in range(TOTAL + 1)
    ]


def test_warmup_phase_increases_linearly() -> None:
    """warmup 段从 0 线性升到 `base_lr`。"""
    lrs = _schedule()[:WARMUP]
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[-1] < 1.0
    for i in range(len(lrs) - 1):
        msg = f"warmup 段应单调上升，第 {i} 步 {lrs[i]} -> {lrs[i + 1]}"
        assert lrs[i] < lrs[i + 1], msg
    expected = [s / WARMUP for s in range(WARMUP)]
    assert lrs == pytest.approx(expected)


def test_cosine_phase_decreases_monotonically() -> None:
    """warmup 结束后按余弦单调下降，起点约为 `base_lr`。"""
    lrs = _schedule()[WARMUP:]
    assert lrs[0] == pytest.approx(1.0)
    for i in range(len(lrs) - 1):
        msg = f"余弦段应单调下降，第 {WARMUP + i} 步 {lrs[i]} -> {lrs[i + 1]}"
        assert lrs[i] > lrs[i + 1], msg


def test_final_lr_hits_min_lr_ratio_floor() -> None:
    """末步恰好落在 `min_lr_ratio × base_lr`，且全程不低于它。"""
    lrs = _schedule(MIN_RATIO)
    final = lrs[-1]
    msg = f"末步 lr 应为 {MIN_RATIO}，实际 {final}"
    assert final == pytest.approx(MIN_RATIO), msg
    # 下界只约束余弦段：warmup 段按定义从 0 起步，不受 min_lr_ratio 影响
    for i, lr in enumerate(lrs[WARMUP:], start=WARMUP):
        msg = f"第 {i} 步 lr {lr} 低于下界 {MIN_RATIO}"
        assert lr >= MIN_RATIO - 1e-12, msg


def test_zero_min_lr_ratio_decays_to_zero() -> None:
    """`min_lr_ratio=0`（默认）时末步衰减到 0。"""
    assert _schedule()[-1] == pytest.approx(0.0)


def test_lambda_lr_matches_warmup_cosine_lr() -> None:
    """`LambdaLR` 每步取值与 `warmup_cosine_lr(step, ...)` 一致（步号从 1 起）。"""
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([param], lr=BASE_LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP, TOTAL)

    observed: list[float] = []
    for _ in range(TOTAL):
        optimizer.step()
        scheduler.step()
        observed.append(float(optimizer.param_groups[0]["lr"]))

    expected = [warmup_cosine_lr(s, BASE_LR, WARMUP, TOTAL) for s in range(1, TOTAL + 1)]
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("warmup_steps", "total_steps", "min_lr_ratio"),
    [(-1, 100, 0.0), (10, 0, 0.0), (100, 100, 0.0), (10, 100, -0.1), (10, 100, 1.5)],
)
def test_invalid_config_raises(warmup_steps: int, total_steps: int, min_lr_ratio: float) -> None:
    """非法步数 / 非法 `min_lr_ratio` 直接报错，不静默产出错误的 lr。"""
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, 1.0, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio)
    with pytest.raises(ValueError):
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([param], lr=BASE_LR)
        get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio
        )
