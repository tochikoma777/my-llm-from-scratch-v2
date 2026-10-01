"""模型配置层。

`GPTConfig` 是 v2 的**唯一配置来源**（Single Source of Truth）。

v1 的历史包袱：配置以字面量字典写在 `if __name__ == "__main__"` 里，
既不能 import 也不能参数化，且同一份 `GPT_CONFIG_124M` 在三个文件里有冲突的
`context_length`（256 / 1024 / 256）。这里改成带类型注解的 dataclass：

- 必填字段无默认值，漏配即报错；
- `qkv_bias` 默认 `True`（BUG-3，对齐 GPT-2 规格 —— Conv1D 恒有 bias）；
- yaml 是唯一的外部入口（`GPTConfig.from_yaml`），脚本不再自带配置。

搬运来源：v1 `src/modules/language_module.py` 消费的 cfg 字典键；
          v1 `module_train.py:530-538` / `module_load_param.py:513-518` /
          `module_fine_tuning.py:429-434`。
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class GPTConfig:
    """GPT-2 架构超参数。

    字段与 v1 cfg 字典一一对应，含义与默认值说明见类注释。

    Attributes:
        vocab_size: 词表大小，GPT-2 BPE 为 50257。
        context_length: 最大上下文长度（位置编码行数），标准 GPT-2 为 1024。
        emb_dim: 嵌入/残差流维度，GPT-2 small 为 768。
        n_layers: TransformerBlock 堆叠层数。
        n_heads: 注意力头数，必须整除 emb_dim。
        drop_rate: Dropout 概率，作用于嵌入、注意力权重与两处残差分支。
        qkv_bias: Q/K/V 线性层是否带 bias；必须为 True 才能加载 GPT-2 权重。
    """

    vocab_size: int
    context_length: int
    emb_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float = 0.1
    qkv_bias: bool = True

    def __post_init__(self) -> None:
        """校验基础约束，尽早失败而不是等到 forward 里崩。"""
        if self.emb_dim % self.n_heads != 0:
            msg = f"emb_dim ({self.emb_dim}) 必须能被 n_heads ({self.n_heads}) 整除"
            raise ValueError(msg)
        if not 0.0 <= self.drop_rate < 1.0:
            msg = f"drop_rate 必须在 [0, 1) 区间内，当前: {self.drop_rate}"
            raise ValueError(msg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GPTConfig:
        """从 yaml 文件构造配置。

        Args:
            path: yaml 路径。顶层键即为本 dataclass 的字段名。

        Returns:
            GPTConfig 实例。

        Raises:
            KeyError: yaml 缺少必填字段。
            TypeError: yaml 含未知字段。
        """
        with Path(path).open(encoding="utf-8") as f:
            raw: dict[str, object] = yaml.safe_load(f)

        known = {field.name for field in fields(cls)}
        required = {
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - set(raw)
        if missing:
            msg = f"{path} 缺少必填字段: {sorted(missing)}"
            raise KeyError(msg)
        unknown = set(raw) - known
        if unknown:
            msg = f"{path} 含未知字段: {sorted(unknown)}，合法字段: {sorted(known)}"
            raise TypeError(msg)

        return cls(**raw)  # type: ignore[arg-type]

    @classmethod
    def gpt2_small(cls) -> GPTConfig:
        """严格对齐 OpenAI GPT-2 small（124M）的预设。

        Returns:
            配置对象：`emb_dim=768, n_layers=12, n_heads=12, context_length=1024, qkv_bias=True`。
        """
        return cls(
            vocab_size=50257,
            context_length=1024,
            emb_dim=768,
            n_layers=12,
            n_heads=12,
            drop_rate=0.0,
            qkv_bias=True,
        )

    @classmethod
    def gpt2_tiny(cls) -> GPTConfig:
        """CPU 上几分钟能跑完的演示配置（`configs/gpt2-tiny.yaml` 的等价物）。

        Returns:
            配置对象：`emb_dim=128, n_layers=2, n_heads=4, context_length=256`。
        """
        return cls(
            vocab_size=50257,
            context_length=256,
            emb_dim=128,
            n_layers=2,
            n_heads=4,
            drop_rate=0.1,
            qkv_bias=True,
        )
