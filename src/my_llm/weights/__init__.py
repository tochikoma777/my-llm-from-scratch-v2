"""权重加载：OpenAI TF 检查点兼容层 + HuggingFace 加载。

两条路并存的原因：
- `openai_tf.py` 是 v1 的独特资产（`module_load_param.py`），能从原始 TensorFlow checkpoint
  还原权重；保留它是为了做**双路交叉验证**——同一次 TF 加载与 HF 加载应得到同一个张量。
- `hf.py` 是 v2 的主力：加载 HF 权重不需要 tensorflow（v1 的硬依赖，见审计 §4.9），
  也是 parity 测试的取数方式。

**重要**：v1 的 `load_weights_into_gpt` 用 `assign()` 把 `wte` 拷给 `out_head`，破坏了权重 tie
（BUG-1）。v2 的 `GPTModel` 已在 `__init__` 里共享了同一个 `nn.Parameter`，
因此两个加载器都**不要**再给 `out_head.weight` 赋值。
"""

from my_llm.weights.hf import gpt_config_from_hf as gpt_config_from_hf
from my_llm.weights.hf import hf_key_to_ours as hf_key_to_ours
from my_llm.weights.hf import load_hf_state_dict as load_hf_state_dict
from my_llm.weights.hf import load_hf_weights_into_gpt as load_hf_weights_into_gpt
from my_llm.weights.hf import load_weights_from_hf as load_weights_from_hf
from my_llm.weights.openai_tf import download_and_load_gpt2 as download_and_load_gpt2
from my_llm.weights.openai_tf import (
    load_openai_tf_weights_into_gpt as load_openai_tf_weights_into_gpt,
)
from my_llm.weights.openai_tf import load_weights_into_gpt as load_weights_into_gpt

__all__ = [
    "download_and_load_gpt2",
    "gpt_config_from_hf",
    "hf_key_to_ours",
    "load_hf_state_dict",
    "load_hf_weights_into_gpt",
    "load_openai_tf_weights_into_gpt",
    "load_weights_from_hf",
    "load_weights_into_gpt",
]
