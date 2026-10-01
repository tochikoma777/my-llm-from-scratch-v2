"""pytest 公共 fixture。

parity 相关约定（见 CODEBUDDY.md 硬约束 2）：
- 所有对比类测试跑 CPU + fp32，阈值 1e-5 不可放宽；
- 需要真实 GPT-2 权重的 fixture（如下一轮的 `hf_gpt2`）必须配 `@pytest.mark.slow` 的用例。

`DEVICE` / `DTYPE` 放在模块级常量而不是 fixture 里，是为了让测试能直接 import 使用，
也方便一眼看出这套测试不碰 CUDA。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel

DEVICE = "cpu"
DTYPE = torch.float32

# 相对仓库根的路径；pytest 从根运行，但这里用 __file__ 定位，避免换 CWD 就崩
REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG_PATH = REPO_ROOT / "configs" / "gpt2-tiny.yaml"


@pytest.fixture(scope="session")
def tiny_cfg() -> GPTConfig:
    """tiny 配置（等价 `GPTConfig.gpt2_tiny()`，唯一来源仍是 yaml）。"""
    return GPTConfig.from_yaml(TINY_CONFIG_PATH)


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg: GPTConfig) -> GPTModel:
    """随机初始化的 tiny 模型：`eval()` + `no_grad`，不含真实权重。

    Args:
        tiny_cfg: `configs/gpt2-tiny.yaml` 读出的配置。

    Returns:
        参数已确定、`requires_grad` 仍为 True 但不做反向的 GPTModel。
    """
    torch.manual_seed(0)
    model = GPTModel(tiny_cfg)
    model.eval()
    model.to(device=DEVICE, dtype=DTYPE)
    return model


@pytest.fixture
def sample_ids() -> torch.Tensor:
    """固定的 token ID 批次 `(2, 16)`，值域 `[0, 10000)`。"""
    torch.manual_seed(0)
    return torch.randint(0, 10000, (2, 16), device=DEVICE)
