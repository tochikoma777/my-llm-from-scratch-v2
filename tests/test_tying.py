"""weight tying 不变量（硬约束 1）。

验证的是**共享同一块内存**，而不只是数值相等 —— v1 的 BUG-1 正是"数值相等但互不相干"，
 tie 一旦被打断，微调时 `out_head` 与 `tok_emb` 会各自漂移，且肉眼无法察觉。
"""

from __future__ import annotations

import torch

from my_llm.model.gpt import GPTModel


def test_head_is_same_parameter(tiny_model: GPTModel) -> None:
    """`out_head.weight` 与 `tok_emb.weight` 必须是同一个 `nn.Parameter` 对象。"""
    is_same = tiny_model.out_head.weight is tiny_model.tok_emb.weight
    msg = "out_head.weight 与 tok_emb.weight 不是同一对象（tie 被打断）"
    assert is_same, msg


def test_head_shares_storage(tiny_model: GPTModel) -> None:
    """同一 `data_ptr()`：写入一侧，另一侧必须同步可见。"""
    a = tiny_model.out_head.weight.data_ptr()
    b = tiny_model.tok_emb.weight.data_ptr()
    msg = f"data_ptr 不一致: out_head={a}, tok_emb={b}"
    assert a == b, msg

    # 反向验证：改 tok_emb 的值，out_head 必须跟着变（单纯的相等关系不会如此）
    with torch.no_grad():
        before = tiny_model.out_head.weight[0, 0].item()
        tiny_model.tok_emb.weight[0, 0] = before + 1.0
        propagated = tiny_model.out_head.weight[0, 0].item()
        tiny_model.tok_emb.weight[0, 0] = before  # 还原，避免污染 session 级 fixture
    assert propagated == before + 1.0, "修改 tok_emb 后 out_head 未同步，说明 ties 已断开"


def test_out_head_has_no_bias(tiny_model: GPTModel) -> None:
    """输出头对齐 HF `lm_head`：无 bias。"""
    bias = tiny_model.out_head.bias
    msg = f"out_head 不应有 bias，当前: {None if bias is None else tuple(bias.shape)}"
    assert bias is None, msg
