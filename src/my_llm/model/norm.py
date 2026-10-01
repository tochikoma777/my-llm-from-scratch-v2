"""归一化与激活函数。

搬运来源（实现逻辑保持不变，只调整组织方式）：
- v1 `src/modules/language_module.py:196` `LayerNorm`
- v1 `src/modules/language_module.py:262` `GELU`

两处命名/说明上的改动：
1. v1 的 `GELU` 实际实现的是 OpenAI GPT-2 的 **tanh 近似**（Hendrycks & Gimpel），
   等价于 transformers 的 `gelu_new`（已实测：`GPT2Config().activation_function == "gelu_new"`，
   其 `forward` 与本实现的系数逐字相同）。它不是 erf 精确版，故更名为 `NewGELU`，避免误读。
2. `LayerNorm` 的 `eps` 由硬编码改为构造参数（默认仍为 1e-5，
   与 HF `layer_norm_epsilon=1e-05` 一致）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """层归一化（LayerNorm，v1 `LM:196`）。

    对最后一个维度做归一化，再施加可学习仿射变换：
    `y = scale * (x - mean) / sqrt(var + eps) + shift`。

    与 `nn.LayerNorm` 对齐的细节：
    - 方差用**有偏**估计（`unbiased=False`，除以 N）—— 与 HF 的 `nn.LayerNorm` 一致；
    - `eps` 默认 1e-5 —— 与 HF `GPT2Config.layer_norm_epsilon` 一致。

    Attributes:
        eps: 数值稳定项。
        scale: 可学习的 gamma，初始化为全 1。
        shift: 可学习的 beta，初始化为全 0。
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        """初始化 LayerNorm。

        Args:
            emb_dim: 归一化所在的特征维度（= GPTConfig.emb_dim）。
            eps: 防止除零的小常数。
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 任意形状 `(*, emb_dim)` 的张量。

        Returns:
            与输入同形状的归一化结果。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class NewGELU(nn.Module):
    """GPT-2 使用的 tanh 近似 GELU（v1 `LM:262`，即 HF 的 `gelu_new`）。

    公式：
        `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`

    与 transformers 的关系（实测 5.12.0）：
    `GPT2Config().activation_function == "gelu_new"`，对应 `NewGELUActivation`，
    其 `forward` 与本实现的常数完全一致，因此本模块与 HF 默认 GPT-2 **数值等价**。
    """

    def __init__(self) -> None:
        """无参数，仅存在于继承 `nn.Module` 以参与图的可视化与设备迁移。"""
        super().__init__()
        self.coef = 0.044715
        self.scale = math.sqrt(2.0 / math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 任意形状张量。

        Returns:
            逐元素 GELU 结果，形状不变。
        """
        return 0.5 * x * (1.0 + torch.tanh(self.scale * (x + self.coef * torch.pow(x, 3))))
