"""my-llm-from-scratch v2。

从零实现的 GPT-2，目标是与 HuggingFace `transformers` 数值对齐（parity）。
架构决策与 v1 审计证据见 `docs/00-现状盘点.md`。

子包：
- `config` — `GPTConfig`，唯一配置来源
- `model` — 归一化 / 注意力 / Transformer 块 / GPTModel（已实现）
- `weights` — OpenAI TF 检查点兼容层 + HuggingFace 加载
- `tokenizer` — Tokenizer 协议与 tiktoken 实现
- `data` — 滑窗数据集与 dataloader
- `train` — loss / 指标 / scheduler / trainer
- `generate` — 采样策略与 KV cache
- `finetune` — Alpaca 指令微调
- `utils` — 随机数种子 / 日志 / 可视化
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("my-llm-from-scratch")
except PackageNotFoundError:  # 未安装（源码直跑）
    __version__ = "2.0.0.dev0"

__all__ = ["__version__"]
