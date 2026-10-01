"""文本生成：采样策略。

搬运来源：v1 `module_load_param.py:366` `generate`（temperature + top-k）与
`generate_text_simple.py:17`（greedy）。

相对 v1 的改动：
- 拆成可独立测试的小函数（`apply_top_k` / `apply_top_p` / `apply_temperature`），
  v1 把这些逻辑全塞在一个 78 行的循环里（`LOAD:408-435`），无法单测;
- 新增 top-p（nucleus）；
- **`generator` 参数**：v1 依赖全局 `torch.manual_seed`（`LOAD:481`），这里允许注入 `Generator`，
  让采样在测试里可复现而不污染全局 RNG；
- 支持 batch > 1（v1 的 `idx_next.item()` 隐含 batch 必须为 1，`LOAD:438`）。

默认**不使用** KV cache；带 cache 的快路径见 `generate.kv_cache`，两条路径的数值一致性
由 `tests/test_kv_cache.py` 保证。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """按温度缩放 logits。

    Args:
        logits: 形状 `(batch, vocab_size)`。
        temperature: > 0 且 != 1 时生效；<= 0 视为贪婪（不缩放）。

    Returns:
        缩放后的 logits。
    """
    raise NotImplementedError


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """把 top-k 之外的 logits 置为 `-inf`（v1 `LOAD:408-418` 的行为）。

    Args:
        logits: 形状 `(batch, vocab_size)`。
        top_k: 保留的候选数；`None` 或 <= 0 表示不筛选。

    Returns:
        掩码后的 logits。
    """
    raise NotImplementedError


def apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    """nucleus 采样：保留累积概率达到 `top_p` 的最小候选集（v2 新增）。

    Args:
        logits: 形状 `(batch, vocab_size)`，通常是**已缩放**的 logits。
        top_p: 累积概率阈值，取值 (0, 1]；`None` 或 >= 1 表示不筛选。

    Returns:
        掩码后的 logits。
    """
    raise NotImplementedError


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从最后一个时间步采样下一个 token。

    Args:
        logits: 形状 `(batch, seq_len, vocab_size)` 或 `(batch, vocab_size)`。
        temperature: 温度；<= 0 时退化为贪婪。
        top_k: top-k 阈值。
        top_p: nucleus 阈值。
        generator: 随机源，便于复现。

    Returns:
        形状 `(batch, 1)` 的 token ID。
    """
    raise NotImplementedError


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """自回归生成（v1 `LOAD:366` 的等价实现，支持 batch > 1）。

    Args:
        model: 语言模型。
        idx: 起始序列，形状 `(batch, seq_len)`。
        max_new_tokens: 新生成的 token 上限。
        context_size: 每次前向截断的上下文长度。
        temperature: 温度；<= 0 走贪婪。
        top_k: top-k 阈值。
        top_p: nucleus 阈值。
        eos_id: 结束符 ID；批内有样本命中时该样本停止（其余继续）。
        generator: 随机源。

    Returns:
        拼接后的完整序列，形状 `(batch, seq_len + generated)`。
    """
    raise NotImplementedError
