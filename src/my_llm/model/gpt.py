"""完整 GPT 模型。

搬运来源：v1 `src/modules/language_module.py:470` `GPTModel`（结构不变，配置改为 `GPTConfig`）。

相对 v1 的关键修复 —— **BUG-1 权重 tie**：

    v1 在 `LOAD:361` 用 `assign()` 把 `wte` 赋给 `out_head.weight`，而 `assign` 返回的是
    `torch.nn.Parameter(torch.tensor(right))`，即**按值拷贝**。于是加载后
    `out_head.weight` 与 `tok_emb.weight` 数值相等却是两个互不相关的张量；一旦微调，
    二者会各自更新并漂移。HF 实测
    `lm_head.weight.data_ptr() == transformer.wte.weight.data_ptr()` 为 True，
    是真正的共享（2026-10-01 审计，transformers 5.12.0）。

    这里在 `__init__` 里直接把同一个 `nn.Parameter` 对象赋给 `out_head.weight`，并用断言锁住不变量。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from my_llm.config import GPTConfig
from my_llm.model.block import TransformerBlock
from my_llm.model.norm import LayerNorm


class GPTModel(nn.Module):
    """纯解码器 GPT（v1 `LM:470`）。

    前向流程：词嵌入 + 可学习位置编码 → Dropout → N × TransformerBlock → final_norm → 词表投影。

    Attributes:
        tok_emb: 词嵌入，`(vocab_size, emb_dim)`。
        pos_emb: 位置嵌入，`(context_length, emb_dim)`，可学习。
        drop_emb: 嵌入后的 Dropout。
        trf_blocks: N 层 `TransformerBlock`。
        final_norm: 输出前的 LayerNorm（对应 HF `transformer.ln_f`）。
        out_head: 词表投影，**无 bias**，权重与 `tok_emb` 共享（tied）。
        cfg: 构造时的配置快照，供权重加载/保存复用以做形状校验。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """初始化模型。

        Args:
            cfg: 模型配置。

        Raises:
            AssertionError: weight tying 未生效（不该发生，仅作为不变量保护）。
        """
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)

        # 输出头无 bias（对齐 HF lm_head）
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # BUG-1：真正的权重绑定 —— 共享同一个 nn.Parameter 对象，不是拷贝。
        self.out_head.weight = self.tok_emb.weight
        # 注：断言消息先落到变量，避免 ruff 0.6.x（pre-commit）与 0.16.x（本地）
        # 对 assert 的换行风格不一致 —— 那样会导致两边互相改写文件
        tie_msg = "out_head.weight 必须与 tok_emb.weight 共享同一个 nn.Parameter"
        assert self.out_head.weight is self.tok_emb.weight, tie_msg

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            in_idx: token ID 张量，形状 `(batch_size, seq_len)`。

        Returns:
            logits，形状 `(batch_size, seq_len, vocab_size)`。
        """
        _batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)

        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # nn.Module.__call__ 在 torch 的类型存根里返回 Any（同上）
        logits: torch.Tensor = self.out_head(x)
        return logits
