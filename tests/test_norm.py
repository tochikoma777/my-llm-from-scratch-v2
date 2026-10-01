"""LayerNorm / NewGELU 与参考实现的数值对齐。

两个对照对象都是本机实测的库实现（不是重写一份公式）：
- LayerNorm → `torch.nn.LayerNorm`（HF 内部用的就是它，`GPT2Block.ln_1` 为 `nn.LayerNorm`）；
- NewGELU → `transformers.activations.NewGELUActivation`（即 `gelu_new`，GPT-2 默认值）。

阈值沿用硬约束 2 的 1e-5（激活函数更严格，取 1e-6），断言消息里带上实际 diff，
失败时无需重跑就能看出是数值漂移还是公式写错。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.activations import NewGELUActivation

from my_llm.config import GPTConfig
from my_llm.model.norm import LayerNorm, NewGELU


@pytest.fixture
def emb_dim(tiny_cfg: GPTConfig) -> int:
    """tiny 配置的 `emb_dim`。"""
    return tiny_cfg.emb_dim


def test_layer_norm_matches_torch(emb_dim: int) -> None:
    """自实现 LayerNorm 与 `nn.LayerNorm` 数值一致（含随机仿射参数）。"""
    torch.manual_seed(0)
    x = torch.randn(2, 16, emb_dim, dtype=torch.float32)

    ours = LayerNorm(emb_dim)
    # 随机化仿射参数，避免全 1 / 全 0 的退化情形掩盖错误
    nn.init.normal_(ours.scale, mean=0.5, std=1.0)
    nn.init.normal_(ours.shift, mean=-0.2, std=0.5)

    ref = nn.LayerNorm(emb_dim, eps=ours.eps)
    with torch.no_grad():
        ref.weight.copy_(ours.scale)
        ref.bias.copy_(ours.shift)

    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x)

    diff = (y_ours - y_ref).abs().max().item()
    msg = (
        f"LayerNorm max abs diff = {diff:.3e}（阈值 1e-5）；"
        f"emb_dim={emb_dim}, x shape={tuple(x.shape)}"
    )
    assert diff < 1e-5, msg
    print(f"[test_layer_norm_matches_torch] {msg}")


def test_new_gelu_matches_transformers() -> None:
    """自实现 NewGELU 与 HF `NewGELUActivation` 数值一致。"""
    torch.manual_seed(0)
    # 特意混入负值、零与较大正值：tanh 近似的非线性区域都覆盖到
    x = torch.cat(
        [
            torch.randn(64) * 3.0,
            torch.zeros(3),
            torch.tensor([-10.0, -1.0, 0.5, 10.0]),
        ]
    )

    ours = NewGELU()
    ref = NewGELUActivation()

    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x)

    diff = (y_ours - y_ref).abs().max().item()
    msg = (
        f"NewGELU max abs diff = {diff:.3e}（阈值 1e-6）；"
        f"n={x.numel()}, x range=[{x.min():.2f}, {x.max():.2f}]"
    )
    assert diff < 1e-6, msg
    print(f"[test_new_gelu_matches_transformers] {msg}")
