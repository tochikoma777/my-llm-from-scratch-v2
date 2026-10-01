"""Transformer 块：前馈网络 + 残差堆叠。

搬运来源（逻辑不变，仅重新组织）：
- v1 `src/modules/language_module.py:305` `FeedForward`
- v1 `src/modules/language_module.py:357` `TransformerBlock`

结构仍是 **Pre-LN + 双层残差**（与 HF `GPT2Block` 一致，见 `docs/00-现状盘点.md` 第二部分）：

    x = x + drop(attn(norm1(x)))
    x = x + drop(ff(norm2(x)))

Dropout 位置说明：HF 用 `attn_pdrop`/`resid_pdrop`/`embd_pdrop` 三个开关，本实现共用一个
`cfg.drop_rate`（沿用 v1 语义）：注意力权重一次、两处残差分支各一次。fp32 数值等价。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from my_llm.config import GPTConfig
from my_llm.model.attention import MultiHeadAttention
from my_llm.model.norm import LayerNorm, NewGELU


class FeedForward(nn.Module):
    """前馈网络（v1 `LM:305`）：`Linear -> NewGELU -> Linear`，中间层 4 倍扩展。

    相对 v1 的结构性改动：`nn.Sequential` 换成**具名层**。
    v1 用 `self.layers[0]` / `self.layers[2]` 取两个线性层（因为 `[1]` 是无参数的 GELU），
    这是隐式下标约定 —— 调整 FFN 结构时必碎。v2 直接按名字访问。

    Attributes:
        fc1: 升维线性层，`emb_dim -> 4 * emb_dim`（HF `mlp.c_fc`）。
        gelu: GELU 激活。
        fc2: 降维线性层，`4 * emb_dim -> emb_dim`（HF `mlp.c_proj`）。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """初始化前馈网络。

        Args:
            cfg: 模型配置，消费 `emb_dim`。
        """
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim)
        self.gelu = NewGELU()
        self.fc2 = nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状 `(batch, tokens, emb_dim)`。

        Returns:
            与输入同形状的张量。
        """
        output: torch.Tensor = self.fc2(self.gelu(self.fc1(x)))
        return output


class TransformerBlock(nn.Module):
    """单个 Transformer 块（v1 `LM:357`）。

    Attributes:
        att: 多头因果自注意力。
        ff: 前馈网络。
        norm1: 注意力前的 LayerNorm（Pre-LN）。
        norm2: FFN 前的 LayerNorm（Pre-LN）。
        drop_shortcut: 施加在两条残差分支输出上的 Dropout。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """初始化块。

        Args:
            cfg: 模型配置。
        """
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状 `(batch, tokens, emb_dim)`。

        Returns:
            与输入同形状的张量。
        """
        # 注意力子层 + 残差
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈子层 + 残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut
