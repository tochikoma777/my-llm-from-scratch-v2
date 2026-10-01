"""多头注意力与 `torch.nn.MultiheadAttention` 的数值对齐。

对照对象：`torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True)`。
它把 Q/K/V 打包进单个 `in_proj_weight`（形状 `(3*embed_dim, embed_dim)`，顺序 **q → k → v**），
而本实现是三个独立的 `nn.Linear`。

因为两边都用 `nn.Linear` 语义的 `(out_features, in_features)` 布局，
所以拼接时**不需要转置**；`out_proj` 同理原样拷贝。
若第 4 步把这里的方向搞反，diff 会落在 1e-2 量级（形状恰好相同，但内容是另一个头），
断言消息里给了这一提示。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from my_llm.model.attention import MultiHeadAttention
from my_llm.model.gpt import GPTModel


def _build_reference(
    ours: MultiHeadAttention, emb_dim: int, num_heads: int
) -> nn.MultiheadAttention:
    """用 our self-attention 的权重填出一个等价的 `nn.MultiheadAttention`。

    Args:
        ours: 本仓库的注意力模块（eval 模式）。
        emb_dim: 嵌入维度。
        num_heads: 头数。

    Returns:
        权重已同步、dropout 关闭的参考实现。
    """
    ref = nn.MultiheadAttention(
        embed_dim=emb_dim, num_heads=num_heads, dropout=0.0, batch_first=True, bias=True
    )
    ref.eval()

    if ours.W_query.bias is None or ours.W_key.bias is None or ours.W_value.bias is None:
        msg = "qkv_bias 必须为 True 才能与 bias=True 的 MultiheadAttention 对齐"
        raise AssertionError(msg)

    with torch.no_grad():
        # in_proj 顺序为 q, k, v；nn.Linear 的 weight 已是 (out, in)，直接 cat 即可
        ref.in_proj_weight.copy_(
            torch.cat([ours.W_query.weight, ours.W_key.weight, ours.W_value.weight], dim=0)
        )
        ref.in_proj_bias.copy_(
            torch.cat([ours.W_query.bias, ours.W_key.bias, ours.W_value.bias], dim=0)
        )
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(ours.out_proj.bias)
    return ref


def test_attention_matches_pytorch_mha(tiny_model: GPTModel) -> None:
    """自实现 MultiHeadAttention 与 `nn.MultiheadAttention` 在因果掩码下数值一致。"""
    torch.manual_seed(0)
    ours = tiny_model.trf_blocks[0].att
    cfg = tiny_model.cfg
    x = torch.randn(2, 8, cfg.emb_dim, dtype=torch.float32)

    ref = _build_reference(ours, cfg.emb_dim, cfg.n_heads)

    num_tokens = x.shape[1]
    # 本实现的 mask：上三角（diagonal=1）为 1.0 表示被屏蔽；torch 的 bool mask True 同义
    attn_mask = ours.mask[:num_tokens, :num_tokens].to(torch.bool)

    with torch.no_grad():
        y_ours = ours(x)
        y_ref, _ = ref(x, x, x, need_weights=False, attn_mask=attn_mask)

    diff = (y_ours - y_ref).abs().max().item()
    msg = (
        f"MultiHeadAttention max abs diff = {diff:.3e}（阈值 1e-5）；"
        f"shape={tuple(x.shape)}, heads={cfg.n_heads}, head_dim={cfg.emb_dim // cfg.n_heads}"
    )
    assert diff < 1e-5, msg
    print(f"[test_attention_matches_pytorch_mha] {msg}")


def test_attention_is_causal(tiny_model: GPTModel) -> None:
    """因果性：改动第 t 个位置之后的输入，不能影响第 t 个位置之前的输出。"""
    torch.manual_seed(0)
    ours = tiny_model.trf_blocks[0].att
    x = torch.randn(1, 6, tiny_model.cfg.emb_dim, dtype=torch.float32)

    # 只改最后两个 token，前四个位置的输出必须逐位不变
    x_perturbed = x.clone()
    x_perturbed[:, 4:, :] += 3.0

    with torch.no_grad():
        y_clean = ours(x)
        y_perturbed = ours(x_perturbed)

    diff = (y_clean[:, :4, :] - y_perturbed[:, :4, :]).abs().max().item()
    msg = f"未来 token 泄漏到过去位置，max abs diff = {diff:.3e}"
    assert diff == 0.0, msg
