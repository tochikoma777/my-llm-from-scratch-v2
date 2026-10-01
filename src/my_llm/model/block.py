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

    Attributes:
        layers: `Sequential(Linear(emb, 4*emb), NewGELU, Linear(4*emb, emb))`。
            注意索引 1 是激活函数，所以线性层在索引 0 与 2 —— 权重映射依赖这一布局
            （v1 `LOAD:321` 与 `LOAD:330`）。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """初始化前馈网络。

        Args:
            cfg: 模型配置，消费 `emb_dim`。
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            NewGELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状 `(batch, tokens, emb_dim)`。

        Returns:
            与输入同形状的张量。
        """
        # nn.Module.__call__ 在 torch 的类型存根里返回 Any（同上）
        output: torch.Tensor = self.layers(x)
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
