# Changelog

本文件记录 v2 相对 v1（`../v1-reference`，my-LLM-from-scratch）的变化。
所有架构结论与行号证据见 [`docs/00-现状盘点.md`](docs/00-现状盘点.md)。

## [2.0.0] — 进行中

### Added

- **工程地基**：`pyproject.toml`（单一依赖来源，含 pytest / ruff(line-length 100) / mypy strict 配置）、
  `Makefile`（install / test / test-full / lint / fmt / check / demo / clean）、
  GitHub Actions（`test` 矩阵 3.10–3.12 + `parity` 仅在 main 触发）、`pre-commit`（ruff + mypy + detect-secrets）、
  `.gitignore`（补齐 v1 遗漏的 `*.pth` `gpt2/` `data/` `outputs/` 等）。
- **配置层**：`GPTConfig` dataclass（`src/my_llm/config.py`）作为唯一配置来源，配套
  `configs/gpt2-small.yaml`（严格对齐 GPT-2 small）与 `configs/gpt2-tiny.yaml`（CPU 演示）。
- **一批 v1 完全缺失的能力**（已建好模块骨架）：
  - `train/scheduler.py`：warmup + cosine（`00-现状盘点.md` §1.2 第 3 项，v1 grep `scheduler` 0 命中）
  - `train/trainer.py`：checkpoint 续训 + 梯度累积 + 混合精度（v1 只有一次性的 `torch.save`，见 §1.2 第 2、4 项）
  - `train/metrics.py`：perplexity（v1 只有交叉熵 loss，见 §1.2 第 5 项）
  - `generate/kv_cache.py`：KV cache（v1 每步全量重算，O(n²)，见 §1.2 第 10 项）
  - `weights/hf.py`：直接加载 HuggingFace 权重，不再强依赖 tensorflow
- **parity 测试基建**：`tests/` 目录 + `slow` marker（`pytest -m slow` 才会下载真实 GPT-2 权重）。

### Changed

- **包结构与导入方式**：扁平脚本（`from language_module import GPTModel`，绑定 CWD）改为
  `src/my_llm/` 正式包 + 类型注解；不再依赖 CWD 相对路径才能跑。
- **模型内核**（从 v1 `language_module.py` 搬运，逻辑不变，仅重新组织到 `model/` 四个文件）：
  `norm.py` / `attention.py` / `block.py` / `gpt.py`。

### Fixed

- **BUG-1 权重 tie**：v1 里 `load_weights_into_gpt` 用 `assign()` 造出的是 **副本**，加载后
  `out_head.weight` 与 `tok_emb.weight` 是两个独立张量，微调会各自更新而漂移；
  HF 实测二者 `data_ptr()` 相同。v2 在 `GPTModel.__init__` 中直接共享同一个 `nn.Parameter` 并加断言。
- **BUG-2 掩码值**：`-torch.inf` → `torch.finfo(dtype).min`，且填在**缩放之后**的分数上，
  与 HF `GPT2Attention` 对齐；同时避免 fp16 下的 `-inf` NaN 风险。
- **BUG-3 `qkv_bias` 默认值**：`False` → `True`。v1 从零训练配置用 `False`，与 GPT-2 规格矛盾，
  且会让 `load_weights_into_gpt` 在给 `bias=None` 赋值时崩溃。
- **NewGELU 命名**：v1 的类名 `GELU` 实际实现的是 OpenAI/GPT-2 的 tanh 近似（= HF `gelu_new`），
  易被误读为 erf 精确版，v2 更名 `NewGELU`。

### Removed

- `archives/`（12 个历史草稿文件，1053 行，与 `src/modules/` 大面积重复且半数是注释掉的死代码）。
- v1 中 `__main__` 内的硬编码配置（`GPT_CONFIG_124M` 等，无法被 import）。
- tensorflow 从主依赖降级为可选项（仅 `weights/openai_tf.py` 兼容层需要）。
