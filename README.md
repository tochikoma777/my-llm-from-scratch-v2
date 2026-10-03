# my-LLM-from-scratch v2

从零实现 GPT-2，并与 HuggingFace `transformers` 做**数值对齐**（numerical parity）。
v2 相对 v1 的全部审计结论与行号证据见 [`docs/00-现状盘点.md`](docs/00-现状盘点.md)。

## 第一屏：parity diff

对照基准：本机实测 **transformers 5.12.0**（`GPT2Config()` 默认值 + `GPT2Block`/`Conv1D` 源码 + `GPT2LMHeadModel.state_dict()` 键名），不是凭记忆。

| 维度 | v1 行为 | HF / OpenAI GPT-2 | v2 状态 | 证据（v1 侧） |
|---|---|---|---|---|
| LayerNorm 位置 | Pre-LN（norm 在 att/ffn 之前） | Pre-LN | ✅ 保持一致 | `LM:439,456` |
| 激活函数 | tanh 近似 GELU（`0.044715`） | `gelu_new`，系数逐字相同 | ✅ 保持一致（更名 `NewGELU`） | `LM:299-302` |
| attn qkv bias | `qkv_bias` 由 config 控制，从零训练默认 **False** | Conv1D 恒有 bias | ✅ 默认改为 True（BUG-3） | `LM:80-82`、`TRAIN:537` |
| attn proj / mlp bias | 默认 True | 有 bias | ✅ 一致 | `LM:86,334,341` |
| 输出头 bias | False | `lm_head` 无 bias | ✅ 一致 | `LM:535` |
| LN eps | 1e-5 | `layer_norm_epsilon=1e-05` | ✅ 一致 | `LM:224` |
| 位置编码 | 可学习 `nn.Embedding`，相加 | `transformer.wpe`，相加 | ✅ 一致 | `LM:514,573` |
| QKV 存储 | `nn.Linear` 存 `(out,in)`，加载时 `.T` | `Conv1D` 存 `(in,out)` | ✅ 数值等价 | `LOAD:283-291` |
| **权重 tie** | **副本**，`out_head` 与 `tok_emb` 是两个张量 | **同一块内存**（`data_ptr` 相等） | ✅ 已修（BUG-1） | `LOAD:361,253` |
| **注意力掩码** | 未缩放分数上填 `-torch.inf` | 缩放后填 `torch.finfo(dtype).min` | ✅ 已修（BUG-2） | `LM:167,171` |
| `context_length` | 三处矛盾：256 / 1024 / 1024 | 1024 | ✅ 统一到 config | `TRAIN:532`、`LOAD:515` |

## Quick start

```bash
make install          # pip install -e ".[dev,viz]" + pre-commit install
make lint             # ruff check + mypy src
make test             # 快测（跳过 @pytest.mark.slow）
make test-full        # parity 测试，会下载 GPT-2 权重
make demo             # tiny 模型训练 + 生成
```

## 项目结构

```
configs/            gpt2-small.yaml（严格对齐 GPT-2 small）/ gpt2-tiny.yaml / sft-alpaca.yaml
src/my_llm/
  config.py         GPTConfig dataclass —— 唯一配置来源
  model/            norm.py · attention.py · block.py · gpt.py（已实现）
  weights/          openai_tf.py（v1 资产移植）· hf.py（新增）
  tokenizer/        protocol.py · tiktoken_impl.py
  data/             dataset.py · dataloader.py
  train/            trainer.py（续训/累积/AMP）· scheduler.py（warmup+cosine）· losses.py · metrics.py（perplexity）
  generate/         sampling.py（top-k/top-p）· kv_cache.py
  finetune/         sft.py（Alpaca）
  utils/            seed.py · logging.py · viz.py
scripts/            train.py · generate.py · sft.py · download_weights.py
```

## 已知状态

- `model/` 四个模块与 `config.py` 已实现；其余模块目前是**类型完整的存根**（`raise NotImplementedError`）。
  填充顺序见 `CODEBUDDY.md` 的「存根填充优先级（P0→P3）」一节（判据是「完成后能做什么」）。
- `tests/` 已有 33 个用例：快测 15 个（`pytest -q`）+ 慢测 18 个（`pytest -m slow`，parity 16 + crossload 2）。
  慢测默认跳过，不联网；`pytest -m slow` 首次会下载 GPT-2 权重。
