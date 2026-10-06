# Changelog

本文件记录 v2 相对 v1（`../v1-reference`，my-LLM-from-scratch）的变化，
格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。
所有架构结论与 `文件:行号` 证据见 [`docs/00-现状盘点.md`](docs/00-现状盘点.md)。

## [2.0.0] — 2026-10-06

v2 不是给 v1 打补丁，而是**完全重写**：从扁平脚本（`from language_module import GPTModel`、
绑死 CWD）改成 `src/my_llm/` 正式包，配置从三处互相矛盾的 `GPT_CONFIG_124M` 字面量
收敛到 `GPTConfig` + yaml，并补上了 v1 完全没有的一套验证能力。

### Added

- **数值对齐验证（parity）**：把同一份 `gpt2`（124M）权重分别装进本仓库 `GPTModel` 与
  HuggingFace `GPT2LMHeadModel`，逐层对拍 logits、12 个 block 的隐状态、注意力权重与嵌入层。
  实测 fp64 绝对差 `2.842e-13`、fp32 相对差 `7.5e-7`（`pytest -m slow -s` 抓取）。
  这是"写对了"的硬证据，不是"生成出来像人话"的软证据。
- **双路径权重加载 + 交叉验证**：`weights/hf.py`（HuggingFace state_dict）与
  `weights/openai_tf.py`（OpenAI 原版 TensorFlow checkpoint）两条独立路径，
  由 `tests/test_crossload.py` 对拍，实测互差精确为 `0`。
- **KV cache**：`generate/kv_cache.py`。实测 CPU 上 1.8x–2.4x 加速，且与无 cache 路径
  **逐 token 完全一致**。实现上不改动 `model/`（用 forward hook 旁路），
  因此不触发"改 `model/` 必须同步 parity"的连锁改动。
- **Alpaca 指令微调**：`finetune/sft.py` + `scripts/sft.py`（变长补齐 + 损失屏蔽，
  prompt 模板与 v1 逐字一致）。权重三态：默认加载 HF `gpt2` / `--checkpoint` 本地 `.pth` /
  `--no-pretrained` 随机初始化。
- **训练工具层**：`train/scheduler.py`（warmup + cosine）、`train/metrics.py`（perplexity）、
  `train/trainer.py`（checkpoint 续训 + 梯度累积 + fp16/bf16/fp32 混合精度）。
- **工具层三件套**：`utils/seed.py`（可复现）、`utils/logging.py`（`my_llm.*` 命名空间）、
  `utils/viz.py`（loss 曲线 + 注意力热力图，只用 matplotlib）。
- **工程地基**：`pyproject.toml`（依赖与 ruff / mypy / pytest 配置单一来源）、`Makefile`、
  GitHub Actions（`test` 矩阵 3.10–3.12 + `parity` 仅 main 触发）、`pre-commit`、
  以及补齐 v1 遗漏的 `.gitignore`（`*.pth` / `gpt2/` / `data/` / `outputs/` 等）。
- **教学 notebook**：`notebooks/01_tokenizer.ipynb`、`02_attention.ipynb`、
  `03_train_and_generate.ipynb`（装 `pip install -e ".[notebooks]"` 后可跑）。

### Changed

- **包结构与导入方式**：扁平脚本改为 `src/my_llm/` 正式包 + 类型注解（`mypy --strict`），
  不再依赖 CWD 相对路径才能跑。
- **配置层拆成并列两份**：`GPTConfig` + `configs/gpt2-*.yaml`（架构）与
  `TrainConfig` + `configs/train-*.yaml`（运行），取代 v1 埋在 `__main__` 里的超参。
- **模型内核重新组织**：v1 `language_module.py` 拆为 `model/` 下的
  `norm.py` / `attention.py` / `block.py` / `gpt.py`，算法逻辑不变。

### Fixed

- **weight tying 被 `assign` 打断**：v1 的 `assign()` 返回新的 `nn.Parameter`，
  加载后 `out_head.weight` 与 `tok_emb.weight` 是两块内存（副本），微调时各自更新而漂移；
  HuggingFace 实测二者 `data_ptr()` 相同。v2 在 `GPTModel.__init__` 共享同一个
  `nn.Parameter`，加载器只 `copy_` 进现有参数，并在结束时断言 tie 仍成立。
- **掩码的持久化与时机**：掩码改为 `register_buffer`（`model/attention.py:78`），
  跟随模块 `.to(device/dtype)` 迁移，不再每步重建；同时填在**缩放之后**的分数上。
- **`qkv_bias` 与 GPT-2 规格不一致**：`False` → `True`。v1 从零训练用 `False`，
  既与 Conv1D 恒有 bias 的规格矛盾，也会让权重加载在给 `bias=None` 赋值时崩溃。
- **混合精度里 bf16 误用 `GradScaler`**：bf16 指数位与 fp32 同宽、不会下溢，不需要 scaler；
  v2 只有 fp16 才启用（`train/trainer.py:249` 的 `enabled=use_scaler`）。
- **`evaluate_model` 的模式污染**：v1 结尾无条件 `model.train()`（`TRAIN:215`），
  评估完把调用方的模型悄悄切回训练模式、dropout 重新打开；
  v2 记录并**恢复进入时的模式**（`train/metrics.py:81-87`）。
- **NewGELU 命名**：v1 的类名 `GELU` 实际实现的是 tanh 近似（= HF `gelu_new`），
  易被误读为 erf 精确版，v2 更名 `NewGELU`。

### Removed

- `archives/`（12 个历史草稿文件，1053 行，与 `src/modules/` 大面积重复且半数是注释掉的死代码）。
- v1 中 `__main__` 内的硬编码配置（`GPT_CONFIG_124M` 等，无法被 import）。
- tensorflow 从主依赖降级为可选项（仅 `weights/openai_tf.py` 兼容层需要）。
