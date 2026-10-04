"""采样：贪婪确定性、batch > 1 独立性、eos per-sample 停止。

除真实 tiny 模型外，还用一个**脚本化假模型**（每步、每行指定下一个 token）——
eos 的 per-sample 行为用随机模型是测不稳的（无法确定它何时命中 eos）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from my_llm.config import GPTConfig
from my_llm.generate.sampling import generate, sample_next_token
from my_llm.model.gpt import GPTModel

VOCAB = 64
EOS = 63  # 必须在词表内，否则假模型的 logits 索引会越界


class ScriptedModel(nn.Module):
    """假模型：第 `step` 步给第 `row` 行指定的 token 打最高分。"""

    def __init__(self, script: list[list[int]]) -> None:
        """初始化。

        Args:
            script: `script[step][row]` = 该步该行要生成的 token ID。
        """
        super().__init__()
        self.script = script
        self.step = 0

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """返回按脚本设定的 logits。

        Args:
            idx: 输入 token ID，形状 `(batch, seq_len)`。

        Returns:
            logits，形状 `(batch, seq_len, VOCAB)`；最后一步只有脚本指定的 token 得分高。
        """
        batch_size, seq_len = idx.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB)
        step = min(self.step, len(self.script) - 1)
        for row, token in enumerate(self.script[step]):
            logits[row, -1, token] = 10.0
        self.step += 1
        return logits


def _tiny_model(tiny_cfg: GPTConfig) -> GPTModel:
    """随机初始化的 tiny 模型（eval 模式）。"""
    torch.manual_seed(0)
    model = GPTModel(tiny_cfg)
    model.eval()
    return model


def test_greedy_is_deterministic(tiny_cfg: GPTConfig) -> None:
    """贪婪解码两次结果完全一致（不依赖 RNG）。"""
    model = _tiny_model(tiny_cfg)
    prompt = torch.randint(0, 1000, (1, 6), dtype=torch.long)
    first = generate(model, prompt, max_new_tokens=8, context_size=tiny_cfg.context_length)
    second = generate(model, prompt, max_new_tokens=8, context_size=tiny_cfg.context_length)
    assert torch.equal(first, second)


def test_greedy_equals_argmax() -> None:
    """`temperature <= 0`（含负数）都退化为 argmax，与 `torch.argmax` 一致。"""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, VOCAB)
    expected = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    assert torch.equal(sample_next_token(logits, temperature=0.0), expected)
    assert torch.equal(sample_next_token(logits, temperature=-1.0), expected)


def test_batch_rows_are_independent(tiny_cfg: GPTConfig) -> None:
    """batch=2 一起生成 == 两行各自单独生成（证明不是 v1 的 batch=1 路径）。"""
    model = _tiny_model(tiny_cfg)
    p1 = torch.randint(0, 1000, (1, 6), dtype=torch.long)
    p2 = torch.randint(1000, 2000, (1, 6), dtype=torch.long)

    batched = generate(
        model, torch.cat((p1, p2), dim=0), 8, tiny_cfg.context_length, temperature=0.0
    )
    single_1 = generate(model, p1, 8, tiny_cfg.context_length, temperature=0.0)
    single_2 = generate(model, p2, 8, tiny_cfg.context_length, temperature=0.0)

    msg = "batch 第 0 行应等于单独生成第 1 条 prompt 的结果"
    assert torch.equal(batched[0], single_1[0]), msg
    assert torch.equal(batched[1], single_2[0])


def test_eos_stops_only_finished_rows() -> None:
    """命中 eos 的行停止（后续继续填 eos），未命中的行照常生成。"""
    script = [[1, 2], [EOS, 3], [4, 5]]
    model = ScriptedModel(script)
    idx = torch.zeros(2, 1, dtype=torch.long)
    out = generate(model, idx, max_new_tokens=3, context_size=32, eos_id=EOS, temperature=0.0)

    assert out[0].tolist() == [0, 1, EOS, EOS], out[0].tolist()
    assert out[1].tolist() == [0, 2, 3, 5], out[1].tolist()


def test_eos_stops_early_when_all_rows_finished() -> None:
    """所有行都命中 eos 后提前结束，不再生成到 `max_new_tokens`。"""
    script = [[EOS, EOS], [7, 7]]
    model = ScriptedModel(script)
    idx = torch.zeros(2, 1, dtype=torch.long)
    out = generate(model, idx, max_new_tokens=10, context_size=32, eos_id=EOS, temperature=0.0)
    assert out.shape[1] == 2, f"应提前停止，实际长度 {out.shape[1]}"


def test_generator_makes_sampling_reproducible() -> None:
    """同一个 `Generator` 两次采样结果一致（v1 依赖全局 RNG，做不到这点）。"""
    torch.manual_seed(0)
    logits = torch.randn(1, 3, VOCAB)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_next_token(logits, temperature=1.0, generator=g1)
    b = sample_next_token(logits, temperature=1.0, generator=g2)
    assert torch.equal(a, b)
