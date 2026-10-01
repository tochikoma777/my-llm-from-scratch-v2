"""OpenAI GPT-2 TensorFlow 检查点 → PyTorch 权重映射（v1 资产移植）。

搬运来源：v1 `src/modules/module_load_param.py`（`download_and_load_gpt2:53`、
`download_file:125`、`load_gpt2_params_from_tf_ckpt:168`、`assign:235`、`load_weights_into_gpt:256`）。

完整键映射见 `docs/00-现状盘点.md` 第三部分。相对 v1 的两处**刻意**改动：

1. **不再给 `out_head.weight` 赋值**。v1 `LOAD:361` 用它/或将 `wte` 拷进去，等于加了
   一份没有被 tie 的副本；v2 的 `GPTModel.__init__` 已共享 `tok_emb.weight`，重复赋值
   会把 tie 重新打断。
2. **tensorflow 改为惰性导入**。它是本模块唯一的可选依赖（`LOAD:24` 在 v1 里是硬 import，
   导致 `--test_mode` 也要装 TF），这里放进函数体内部 import。

保留的语义（不得改动）：
- QKV 按 **q, k, v** 顺序从合并矩阵尾部切分（与 HF `split(split_size, dim=2)` 一致）；
- 所有 Conv1D 形态的权重必须 `.T` 后才对得上 `nn.Linear` 的 `(out, in)`；
- LayerNorm 的 `g/b` 映射到 `scale/shift`。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from my_llm.model.gpt import GPTModel

# 官方权重所在目录；HF 镜像相关环境变量见 Makefile
DEFAULT_BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
ALLOWED_SIZES = ("124M", "355M", "774M", "1558M")
CHECKPOINT_FILENAMES = (
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
)


def download_file(url: str, destination: str | Path) -> None:
    """带进度条的单文件下载（v1 `LOAD:125`）。

    Args:
        url: 远端 URL。
        destination: 本地保存路径；已存在且大小一致时跳过。
    """
    raise NotImplementedError


def download_and_load_gpt2(
    model_size: str, models_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """下载并解析指定尺寸的 GPT-2 权重（v1 `LOAD:53`）。

    Args:
        model_size: `"124M" | "355M" | "774M" | "1558M"`。
        models_dir: 本地目录，实际落在 `<models_dir>/<model_size>/`。

    Returns:
        `(settings, params)`：`settings` 来自 `hparams.json`，`params` 是嵌套字典形式的权重。

    Raises:
        ValueError: `model_size` 不在 `ALLOWED_SIZES` 内。
    """
    raise NotImplementedError


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """把 TF 检查点读成嵌套字典（v1 `LOAD:168`）。

    变量名形如 `model/h0/attn/c_attn/w`，去掉 `model/` 后按 `/` 拆层，
    `h<n>` 对应 `params["blocks"][n]`。

    Args:
        ckpt_path: TF 检查点前缀路径。
        settings: 超参字典，需要 `n_layer`。

    Returns:
        嵌套字典：`{"wpe", "wte", "blocks": [{attn, mlp, ln_1, ln_2}], "g", "b"}`。
    """
    raise NotImplementedError


def assign(left: torch.Tensor, right: npt.NDArray[np.float32]) -> nn.Parameter:
    """形状校验后把 numpy 数组包成 `nn.Parameter`（v1 `LOAD:235`）。

    Args:
        left: 目标参数，**只用于取形状**。
        right: 源 numpy 数组。

    Returns:
        新的 `nn.Parameter`。

    Raises:
        ValueError: 形状不一致。
    """
    raise NotImplementedError


def load_weights_into_gpt(gpt: GPTModel, params: Mapping[str, Any]) -> None:
    """把 TF 参数字典写入 `GPTModel`（v1 `LOAD:256`，去掉了 out_head 赋值）。

    Args:
        gpt: 目标模型。
        params: `load_gpt2_params_from_tf_ckpt` 的输出。

    Raises:
        ValueError: 任一处形状不匹配（由 `assign` 抛出）。
    """
    raise NotImplementedError
