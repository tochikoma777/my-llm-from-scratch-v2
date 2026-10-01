"""多头因果自注意力。

搬运来源：v1 `src/modules/language_module.py:29` `MultiHeadAttention`（实现逻辑不变）。

相对 v1 的唯一**数值**改动是 BUG-2 修复（见下），其余为组织方式与类型注解。

BUG-2（掩码值与填充时机）：
    v1 在**未缩放**的 `Q @ K^T` 上填 `-torch.inf`（`LM:167`），之后才除以
    `sqrt(head_dim)`（`LM:171`）。HF `GPT2Attention` 是**先缩放**、再填 `torch.finfo(dtype).min`。
    两者在 fp32 下 softmax 结果等价，但 fp16 下 `-inf` 参与缩放运算有溢出/NaN 风险，
    且与 HF 不完全同构（parity 测试要求 bit-wise 对齐）。
    故改为：`scores = (Q @ K^T) / sqrt(head_dim)`，然后用 `finfo(dtype).min` 填掩码，最后 softmax。

另一处（非数值）调整：原实现用 `masked_fill_` 原地修改，这里改用非原地版本 `masked_fill`，
避免就地改动参与反向的张量。
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from my_llm.config import GPTConfig


class MultiHeadAttention(nn.Module):
    """多头因果自注意力（v1 `LM:29`）。

    `Attention(Q, K, V) = softmax(QK^T / sqrt(head_dim)) V`，并用上三角掩码保证自回归因果性。

    存储约定与 HF 的差异（数值等价，见 `docs/00-现状盘点.md` 第二部分）：
    - 本实现用 `nn.Linear`，权重形状 `(out_features, in_features)`；
    - HF 的 `Conv1D` 存的是转置形态 `(in_features, out_features)`；
    - 二者通过加载时的 `.T`（`weights/openai_tf.py`）对齐，Q/K/V 的切分顺序也一致（q, k, v）。

    Attributes:
        d_out: 输出维度（= emb_dim）。
        num_heads: 头数。
        head_dim: 每头维度 `d_out // num_heads`。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """初始化注意力模块。

        Args:
            cfg: 模型配置，消费 `emb_dim` / `context_length` / `n_heads` / `drop_rate` /
                `qkv_bias`。
        """
        super().__init__()
        # 保证每个头的维度为整数
        if cfg.emb_dim % cfg.n_heads != 0:
            msg = (
                f"emb_dim ({cfg.emb_dim}) 必须能被 n_heads ({cfg.n_heads}) 整除，"
                f"否则 head_dim 非整数: {cfg.emb_dim}/{cfg.n_heads}"
            )
            raise ValueError(msg)

        self.d_out = cfg.emb_dim
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads

        # Q/K/V 投影：bias 由 cfg.qkv_bias 控制，加载 GPT-2 权重时必须为 True（BUG-3）
        self.W_query = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)

        # 多头拼接后的输出投影（Conv1D c_proj 对应物，恒带 bias）
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)

        self.dropout = nn.Dropout(cfg.drop_rate)

        # 因果掩码注册为 buffer：非参数，但会随 state_dict 一起保存/加载
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状 `(batch_size, num_tokens, emb_dim)` 的输入。

        Returns:
            形状 `(batch_size, num_tokens, emb_dim)` 的注意力输出。
        """
        batch_size, num_tokens, _emb_dim = x.shape

        # (batch, tokens, emb_dim) -> (batch, heads, tokens, head_dim)
        keys = (
            self.W_key(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        queries = (
            self.W_query(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.W_value(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 缩放后的注意力分数: (batch, heads, tokens, tokens)
        attn_scores = (queries @ keys.transpose(2, 3)) / (self.head_dim**0.5)

        # BUG-2：先缩放、再按 finfo(dtype).min 填掩码（对齐 HF GPT2Attention）
        # 注：用 .to(torch.bool) 而非 .bool()，后者会让 mypy strict 判定 '"Tensor" not callable'
        # 注：`__ne__`/`to`/`bool` 在 torch 存根里都不是纯 Tensor 返回类型，需要显式 cast
        causal_mask = cast(torch.Tensor, self.mask != 0)
        mask_bool = causal_mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch, heads, tokens, head_dim) -> (batch, tokens, emb_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        # nn.Module.__call__ 在 torch 的类型存根里返回 Any，显式标注后再 return 才过得去 mypy strict
        output: torch.Tensor = self.out_proj(context_vec)
        return output
