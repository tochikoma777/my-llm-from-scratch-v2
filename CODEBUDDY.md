# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Repository state

`v2-work/` 是 v2 重写的工作区；`../v1-reference/`（my-LLM-from-scratch，tochikoma777）是**只读参考**（见下方硬约束 3）。

截至 2026-10-01：仓库有 1 次提交 `398964d feat(R2): project scaffold with model core migrated from v1`，
共 50 个被跟踪文件。已落地的是「工程地基 + 模型内核 + 配置层」，其余模块是**类型完整的存根**。

**已实现**（可运行）：
- `src/my_llm/config.py` — `GPTConfig` dataclass，唯一配置来源；`from_yaml` / `gpt2_small()` / `gpt2_tiny()`
- `src/my_llm/model/` — `norm.py`（LayerNorm + NewGELU）、`attention.py`（MultiHeadAttention）、
  `block.py`（FeedForward + TransformerBlock）、`gpt.py`（GPTModel，含 weight tying）
- `src/my_llm/tokenizer/` — `protocol.py`（Tokenizer 协议）、`tiktoken_impl.py`（tiktoken 实现）

**存根**（函数体 `raise NotImplementedError`，共 14 个文件）：
`data/dataset.py`、`data/dataloader.py`、`finetune/sft.py`、`generate/sampling.py`、`generate/kv_cache.py`、
`train/losses.py`、`train/metrics.py`、`train/scheduler.py`、`train/trainer.py`、`utils/seed.py`、
`utils/logging.py`、`utils/viz.py`、`weights/hf.py`、`weights/openai_tf.py`

`tests/` 目录存在但**为空**（测试待补），`notebooks/` 只有 `.gitkeep`。
`docs/00-现状盘点.md` 是 v1 的完整审计报告（含行号证据、权重映射表、与 HF 的架构差异、取舍建议），
动手前优先读它，不要凭记忆重写结论。

## Environment

- Python 3.11.15 / torch 2.12.0+cu130。**本机有 GPU**（RTX 5060 Laptop，8GB），
  但 parity 测试强制 CPU + fp32（硬约束 2），不要因为看到 CUDA 可用就改用 GPU。
- `transformers` 已装（仅 parity 用）；`tensorflow` **未装**——它只是 `weights/openai_tf.py` 的可选依赖，
  不要把它加回主依赖。
- 网络：PyPI 走 tuna 镜像可通；**github.com 不可达**。因此
  - `pre-commit` 无法拉取新的 hook 版本（`rev` 改不动，改了会卡死在初始化）；
  - parity 测试下载 HF 权重依赖 `Makefile` 里的 `HF_ENDPOINT=https://hf-mirror.com`，不要删掉。

## Commands

```bash
make install     # pip install -e ".[dev,viz]" + pre-commit install
make lint        # ruff check src tests scripts + mypy src（当前全绿）
make fmt         # ruff format src tests scripts
make test        # pytest -q（注意：tests/ 为空，当前以 exit code 5「no tests ran」结束，非故障）
make test-full   # pytest -q -m slow -v，parity 测试，会下载 GPT-2 权重
make check       # lint + test
make demo        # tiny 配置训练 + 生成
make clean       # 清缓存
```

补充约定：
- pytest 默认 `addopts = "-m 'not slow'"`，即日常不跑联网测试；**任何会下载真实权重的测试必须标 `@pytest.mark.slow`**。
- 包通过 editable 安装导入（`import my_llm`）；脚本 `scripts/*.py` 从**仓库根目录**运行，
  `--config` 等路径是相对根目录的（如 `configs/gpt2-tiny.yaml`）。
- `pyproject.toml` 里 `[tool.hatch.build.targets.wheel] packages = ["src/my_llm"]` 是必需的：
  hatchling 按项目名推断的是 `src/my_llm_from_scratch/`，删掉会让 `pip install -e .` 失败。

## Architecture

按层划分，越靠下越基础；当前只有 `config` + `model` + `tokenizer` 有真实实现：

| 包 | 职责 | v1 来源 |
|---|---|---|
| `config.py` | `GPTConfig`（vocab_size / context_length / emb_dim / n_layers / n_heads / drop_rate / qkv_bias），`qkv_bias` 默认 True | v1 三处冲突的 `GPT_CONFIG_124M` |
| `model/` | 模型本体。`norm.py` / `attention.py` / `block.py` / `gpt.py` | `language_module.py` |
| `weights/` | `openai_tf.py`（TF→PT 映射，v1 独特资产）、`hf.py`（HF 加载，parity 取数） | `module_load_param.py` |
| `tokenizer/` | `protocol.py` + `tiktoken_impl.py`（`bpe.py` 留作 P3 可选） | 四处重复的 `tiktoken.get_encoding("gpt2")` |
| `data/` | 滑窗数据集 / dataloader | `data_preprocess.py` |
| `train/` | `losses.py` / `metrics.py`(perplexity) / `scheduler.py`(warmup+cosine) / `trainer.py`(续训+累积+AMP) | `module_train.py` |
| `generate/` | `sampling.py`(greedy/temp/top-k/top-p) / `kv_cache.py` | `module_load_param.py:generate`、`generate_text_simple.py` |
| `finetune/` | Alpaca SFT | `module_fine_tuning.py` |
| `utils/` | `seed.py` / `logging.py` / `viz.py` | v1 无对应 |

模型侧已确认的不变量（改动后必须仍然成立）：
- `model/gpt.py:65` `self.out_head.weight = self.tok_emb.weight` —— 同一 `nn.Parameter` 对象；
- `model/attention.py:116` 掩码用 `torch.finfo(dtype).min`，且填在**缩放之后**的分数上；
- `config.py:48` `qkv_bias: bool = True`。

## Conventions and gotchas

- **中文优先**：注释、docstring、日志与参数帮助文字均用中文，标识符用 ASCII。与既有代码保持一致。
- **配置只能来自 yaml**：超参进 `GPTConfig` 或 `configs/*.yaml`，不要写进 `__main__`（硬约束 4）。
- **ruff 版本分歧**：本地 ruff 0.16.9，pre-commit 里锁的是 v0.6.9，两者对多行 `assert` 的换行风格结论相反，
  会互相改写文件（已踩过一次，导致提交反复失败）。规避办法：断言消息先赋变量、写成单行；
  **不要**去改 hook 的 `rev`（GitHub 不可达，改了装不上）。
- **pre-commit 的 mypy 跑在隔离环境**：`additional_dependencies` 必须显式列出 `torch, numpy, types-PyYAML`，
  否则 `config.py` 的 `import yaml` 会报 `import-untyped`（本地能过是因为本地装了 pyyaml 本体）。
- **产物不要入库**：`.gitignore` 已覆盖 `*.pt *.pth *.ckpt *.safetensors`、`gpt2/ models/ checkpoints/`、
  `data/raw/ data/processed/`、`outputs/ runs/`、`*.pdf`。
- 提交前无需手动格式化：pre-commit 会跑 ruff / ruff-format / mypy / detect-secrets，
  hook 改写文件后**重跑一次提交**即可（第一次失败属正常）。

## 项目硬约束（v2 重构）

1. **weight tying**：任何权重加载代码都不得给 `out_head` 赋值。
   weight tying 由 `GPTModel.__init__` 保证（`self.out_head.weight = self.tok_emb.weight`），
   任何形式的赋值（含 copy / assign / deepcopy）都会重新打断 tie，导致微调时两个头漂移。

2. **parity 测试规格**：强制 `device="cpu"`、`dtype=torch.float32`。
   阈值 `1e-5` 是硬线，任何情况下不得放宽。
   原因：GPU 上 fp32 矩阵乘法有非确定性，噪声与阈值同量级，会产生"有时过有时不过"的幽灵失败。

3. **v1 只读**：`../v1-reference` 是只读参考，任何情况下不得修改其中的文件。

4. **配置唯一来源**：所有超参必须来自 `configs/*.yaml`。
   禁止在 `if __name__ == "__main__"` 里硬编码配置——这是 v1 最大的结构性缺陷，不得重现。
