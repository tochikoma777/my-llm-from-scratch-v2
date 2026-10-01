"""与 HuggingFace `GPT2LMHeadModel` 的端到端数值对齐（parity）。

对照对象是**真实权重**下的 `transformers` 实现，不是重写一份公式：同一份 `gpt2`（124M）
权重分别装进 `GPTModel`（`our_gpt2_loaded`）与 HF 模型（`hf_gpt2`），再逐步比对中间结果。
键名映射、转置方向、QKV 顺序任何一处写错，都会在这里现形。

**两条断言并存**（`_assert_parity`），这是实测后定下来的，不是随手放宽：

- **fp64 / 绝对 `1e-5`**：两边都转 fp64 再比。实测互差 ~3e-13，即计算图与 HF 完全同构。
  这条是"数学正确性"的硬证据，阈值就是硬约束里的 1e-5，没有放宽。
- **fp32 / `1e-5 × max(1, max|ref|)`**：保留真实推理 dtype。GPT-2 残差流的量级到 3e3，
  fp32 相对误差 ~1e-7 → 绝对误差 ~2e-4，**高于 1e-5**；且"我们自己 vs 自己"的
  fp32/fp64 自噪声与互差同量（logits：自噪声 1.2e-4 / 1.1e-4，互差 6.9e-5），
  说明那是精度地板而非实现差异。故按张量量级缩放阈值（等价相对 1e-5），
  真实 bug 依旧会被抓住（见下方量级诊断表）。

diff 量级诊断（失败时先看量级再定位）：

    0 ~ 1e-6    正常 fp32 噪声
    ~1e-5       忘记 eval()（dropout 没关）
    ~1e-3       GELU 变体不对，或掩码用了 -inf
    1e-2~1e-1   漏了转置，或 QKV 切分顺序错
    >=1e0       权重整体没加载

全部用例标 `pytest.mark.slow`（会下载权重，日常 `pytest` 必须跳过）。
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from my_llm.model.block import TransformerBlock
from my_llm.model.gpt import GPTModel

pytestmark = pytest.mark.slow

ATOL = 1e-5
N_BLOCKS = 12


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个张量的最大绝对差。"""
    return (a - b).abs().max().item()


def _assert_parity(
    ours32: torch.Tensor,
    theirs32: torch.Tensor,
    ours64: torch.Tensor,
    theirs64: torch.Tensor,
    label: str,
) -> None:
    """fp64 绝对阈值 + fp32 按量级缩放阈值，两条都要过。

    Args:
        ours32: 本仓库 fp32 结果。
        theirs32: HF fp32 结果。
        ours64: 本仓库 fp64 结果。
        theirs64: HF fp64 结果。
        label: 断言消息里用的名字，如 `"logits"`。

    Raises:
        AssertionError: 任一精度下超出阈值；消息里带实际 diff 数值。
    """
    diff64 = _max_diff(ours64, theirs64)
    diff32 = _max_diff(ours32, theirs32)
    # fp32 阈值按参考张量的量级缩放：等价"相对 1e-5，但 O(1) 量级的张量仍按绝对 1e-5"
    scale = max(1.0, theirs32.abs().max().item())
    tol32 = ATOL * scale

    msg = (
        f"{label}: fp64 diff={diff64:.3e}（绝对阈值 {ATOL:.0e}）; "
        f"fp32 diff={diff32:.3e}（阈值 {tol32:.3e} = {ATOL:.0e}×max|ref| {scale:.1f}）; "
        f"shape={tuple(ours32.shape)}"
    )
    assert diff64 < ATOL, msg
    assert diff32 < tol32, msg
    print(f"[parity] {msg}")


def _our_hidden_states(model: GPTModel, ids: torch.Tensor) -> list[torch.Tensor]:
    """逐层隐状态，下标 0 是嵌入输出，下标 `b + 1` 是第 `b` 个 block 的输出。

    Args:
        model: 本仓库模型（eval 模式）。
        ids: token ID 批次 `(batch, seq_len)`。

    Returns:
        长度 `n_layers + 1` 的列表。
    """
    seq_len = ids.shape[1]
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(seq_len, device=ids.device))
    states = [x]
    for block in model.trf_blocks:
        x = block(x)
        states.append(x)
    return states


def _hf_block_outputs(hf_model: Any, ids: torch.Tensor) -> list[torch.Tensor]:  # noqa: ANN401
    """用 forward hook 抓 HF 每个 block 的输出。

    不能直接吃 `output_hidden_states=True` 的 `hidden_states`：transformers 5.x 会把
    最后一项替换成 `last_hidden_state`（也就是 `ln_f` 之后的结果，不是第 11 个 block 的输出），
    用它对拍会得到 1e0 量级的假失败。hook 拿到的是 block 的原始输出。

    Args:
        hf_model: `GPT2LMHeadModel`。
        ids: token ID 批次。

    Returns:
        12 个 block 的输出，顺序即层序。
    """
    captured: list[torch.Tensor] = []
    handles = []

    def _capture(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        captured.append(output)

    for block in hf_model.transformer.h:
        handles.append(block.register_forward_hook(_capture))
    try:
        with torch.no_grad():
            hf_model(ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _our_attn_weights(block: TransformerBlock, x_in: torch.Tensor, num_heads: int) -> torch.Tensor:
    """重算本仓库某层的注意力权重（softmax 之后），与 HF 的 `attentions` 对拍。

    `MultiHeadAttention.forward` 不返回注意力权重，这里按同一步骤重算：
    `norm1 -> Q/K 投影 -> 分头 -> 缩放 -> 因果掩码 -> softmax`。

    Args:
        block: 目标层。
        x_in: 进入该层的隐状态。
        num_heads: 头数。

    Returns:
        注意力权重，形状 `(batch, num_heads, seq_len, seq_len)`。
    """
    batch_size, num_tokens, _emb_dim = x_in.shape
    head_dim = block.att.head_dim

    x_norm = block.norm1(x_in)
    queries = block.att.W_query(x_norm).view(batch_size, num_tokens, num_heads, head_dim)
    keys = block.att.W_key(x_norm).view(batch_size, num_tokens, num_heads, head_dim)

    scores = (queries.transpose(1, 2) @ keys.transpose(1, 2).transpose(2, 3)) / (head_dim**0.5)
    causal = torch.triu(
        torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=scores.device), diagonal=1
    )
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


def test_full_model_logits(
    our_gpt2_loaded: GPTModel,
    hf_gpt2: Any,  # noqa: ANN401  # transformers 无类型存根
    parity_fp64: tuple[Any, GPTModel],  # noqa: ANN401
    sample_ids: torch.Tensor,
) -> None:
    """整模型 logits 对齐：最强的端到端断言。"""
    hf64, ours64 = parity_fp64
    with torch.no_grad():
        ours32 = our_gpt2_loaded(sample_ids)
        theirs32 = hf_gpt2(sample_ids).logits
        ours_64 = ours64(sample_ids)
        theirs_64 = hf64(sample_ids).logits

    _assert_parity(ours32, theirs32, ours_64, theirs_64, "logits")


@pytest.mark.parametrize("layer_idx", list(range(N_BLOCKS)))
def test_block_hidden_states(
    our_gpt2_loaded: GPTModel,
    hf_gpt2: Any,  # noqa: ANN401
    parity_fp64: tuple[Any, GPTModel],  # noqa: ANN401
    sample_ids: torch.Tensor,
    layer_idx: int,
) -> None:
    """逐 block 比对隐状态，用于二分定位是哪一层开始分叉。"""
    hf64, ours64 = parity_fp64
    with torch.no_grad():
        theirs32 = _hf_block_outputs(hf_gpt2, sample_ids)[layer_idx]
        ours32 = _our_hidden_states(our_gpt2_loaded, sample_ids)[layer_idx + 1]
        theirs_64 = _hf_block_outputs(hf64, sample_ids)[layer_idx]
        ours_64 = _our_hidden_states(ours64, sample_ids)[layer_idx + 1]

    _assert_parity(ours32, theirs32, ours_64, theirs_64, f"block {layer_idx:>2} 隐状态")


@pytest.mark.parametrize("layer_idx", [0, N_BLOCKS - 1])
def test_attention_weights(
    our_gpt2_loaded: GPTModel,
    hf_gpt2: Any,  # noqa: ANN401
    sample_ids: torch.Tensor,
    layer_idx: int,
) -> None:
    """注意力权重对齐：验证 Q/K 投影、缩放与掩码时机。

    只在 fp32 下断言且用**绝对** 1e-5：权重本身是概率、量级 O(1)，噪声地板远低于阈值
    （实测 3.6e-7 ~ 4.2e-6）。HF 侧需要 eager 实现才肯吐出 `attentions`（见 conftest）。
    """
    block = cast(TransformerBlock, our_gpt2_loaded.trf_blocks[layer_idx])
    with torch.no_grad():
        theirs = hf_gpt2(sample_ids, output_attentions=True).attentions[layer_idx]
        hidden_in = _our_hidden_states(our_gpt2_loaded, sample_ids)[layer_idx]
        ours = _our_attn_weights(block, hidden_in, our_gpt2_loaded.cfg.n_heads)

    diff = _max_diff(ours, theirs)
    msg = f"block {layer_idx:>2} 注意力权重: fp32 diff={diff:.3e}（绝对阈值 {ATOL:.0e}）"
    assert diff < ATOL, msg
    print(f"[parity] {msg}; shape={tuple(ours.shape)}")


def test_embeddings(
    our_gpt2_loaded: GPTModel,
    hf_gpt2: Any,  # noqa: ANN401
) -> None:
    """嵌入层数值与 weight tying 语义对齐（权重是 copy_ 进去的，这里应为精确 0）。"""
    with torch.no_grad():
        tok_diff = _max_diff(our_gpt2_loaded.tok_emb.weight, hf_gpt2.transformer.wte.weight)
        pos_diff = _max_diff(our_gpt2_loaded.pos_emb.weight, hf_gpt2.transformer.wpe.weight)

    tok_msg = f"tok_emb.weight: diff={tok_diff:.3e}（阈值 {ATOL:.0e}）"
    assert tok_diff < ATOL, tok_msg
    pos_msg = f"pos_emb.weight: diff={pos_diff:.3e}（阈值 {ATOL:.0e}）"
    assert pos_diff < ATOL, pos_msg

    # 硬约束 1：out_head 与 tok_emb 必须是同一块内存（HF 侧 lm_head/wte 同理）
    tie_msg = "out_head.weight 必须与 tok_emb.weight 共享同一块内存（weight tying）"
    assert (
        our_gpt2_loaded.out_head.weight.data_ptr() == our_gpt2_loaded.tok_emb.weight.data_ptr()
    ), tie_msg

    print(f"[parity] {tok_msg}; {pos_msg}")
