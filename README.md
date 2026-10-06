# my-LLM-from-scratch v2

从零实现 GPT-2，并用测试证明它与 HuggingFace 的数值对齐。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![CI](https://github.com/tochikoma777/my-llm-from-scratch-v2/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-105%20passed-brightgreen)

---

## ⭐ 数值对齐：凭什么说"我写对了"

同一份 `gpt2`（124M）权重分别装进本仓库的 `GPTModel` 和 HuggingFace 的 `GPT2LMHeadModel`，
再逐步比对中间结果——键名映射、转置方向、QKV 切分顺序任何一处写错，都会在这些数字上现形。

| 验证项 | fp64 绝对误差 | fp32 相对误差 |
|---|---|---|
| 全模型 logits（本仓库 `GPTModel` vs HF `GPT2LMHeadModel`） | **2.842e-13** | **7.5e-7** |
| OpenAI TF checkpoint vs HuggingFace 权重（双路径交叉验证） | **0.000e+00** | **0.000e+00** |
| KV cache 加速（tiny / CPU / 60 token 贪婪解码） | — | **1.8x – 2.4x** |

数字怎么来的（`pytest -m slow -s` 实测抓取，不是估算）：

- logits 那一行：fp64 下最大绝对差 `2.842e-13`；fp32 下 `9.155e-05`，参考张量量级 `121.6`，
  即相对 `7.5e-7`。fp32 的 `1e-4` 量级是 fp32 自身的精度地板而非实现差异——
  "我们自己 vs 自己"的 fp32/fp64 自噪声也在同一量级。
- 交叉验证那一行有两条独立路径：`test_tf_and_hf_agree`（真实 OpenAI TF 检查点，需 475MB 下载）
  与 `test_openai_tf_loader_matches_hf_loader`（把 HF state_dict 摆成 TF 参数形态喂给 TF 加载器，
  不需要装 TF），两条都精确为 `0`。
- KV cache 那一行是耗时比，单次数字随机器负载浮动（实测见过 1.79x 与 2.38x），
  但两条路径的输出**逐 token 完全一致**（`test_cache_matches_plain_generation` 守着）。

---

## 为什么又一个 from-scratch 项目

因为 99% 的同类仓库只做到「生成出来的话看着像人话」——而这句话任何 bug 都能满足：
QKV 切分顺序错了、转置方向错了、掩码填在缩放之前、weight tying 悄悄断掉，模型照样能吐出通顺的英文。
**"看着对"和"写对了"是两回事。**

v2 的定位是补上这一步：不只实现，还要**证明**。做法是把自己的实现和 HuggingFace 的
`GPT2LMHeadModel` 放在一起，用同一份真实权重逐层对拍（logits、12 个 block 的隐状态、
注意力权重、嵌入层），并保留两条互相独立的权重加载路径彼此验证。上面那张表就是跑出来的结果。

---

## Quick Start

```bash
git clone https://github.com/tochikoma777/my-llm-from-scratch-v2.git
cd my-llm-from-scratch-v2
python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"                              # 1. 安装
make test                                            # 2. 验证：83 个快测应全绿
python scripts/generate.py --config configs/gpt2-small.yaml \
    --prompt "Every effort moves you"                # 3. 生成（首次会下载 gpt2 权重 ~548MB）
```

第 3 条不给 `--checkpoint` 时会自动拉 HuggingFace 的 `gpt2` 权重；国内网络建议先
`export HF_ENDPOINT=https://hf-mirror.com`（`Makefile` 已为 `make` 目标设了默认值）。
想跑教学 notebook，再装一个 `pip install -e ".[notebooks]"`。

### 训练：**请先自备语料**

仓库里**不带任何数据文件**（`data/raw/` 已被 `.gitignore` 覆盖），训练需要自己指定语料：

```bash
python scripts/train.py --config configs/gpt2-tiny.yaml \
    --train-config configs/train-demo.yaml \
    --data /path/to/your/corpus.txt
```

`--data` 默认是 `data/raw/the-verdict.txt`；把语料放到这个路径后，`make demo`
（tiny 配置训练 1 轮 + 用训完的 checkpoint 生成）就能直接跑。语料不存在时脚本会明确报
`FileNotFoundError: 语料不存在`，不会静默失败。

架构与训练超参是**两份 yaml**：`configs/gpt2-*.yaml`（架构）+ `configs/train-*.yaml`（运行）。
`--epochs / --batch-size / --lr / --seed` 是可选覆盖，不给就回落 yaml。

---

## 功能清单

| 能力 | 入口 | 说明 |
|---|---|---|
| 预训练 | `scripts/train.py` | 滑窗数据集 + warmup/cosine 调度 + 梯度累积 + 混合精度（fp16/bf16/fp32）+ checkpoint 续训 |
| 生成 | `scripts/generate.py` | 贪婪 / temperature / top-k / top-p；可选 KV cache 加速 |
| 指令微调（Alpaca） | `scripts/sft.py` | 变长补齐 + 损失屏蔽；模型回复写回 json |
| 权重加载（双路径） | `my_llm/weights/` | HuggingFace state_dict 与 OpenAI 原版 TensorFlow checkpoint，两条路径数值等价 |
| 评估 | `my_llm/train/metrics.py` | perplexity |
| 可复现与可视化 | `my_llm/utils/` | seed / logging / loss 曲线与注意力热力图 |

几个容易踩的点：

- **SFT 的权重三态**：缺省加载 HF `gpt2` 预训练权重（指令微调的前提）→ `--checkpoint <本地 .pth>`
  → `--no-pretrained` 才随机初始化（仅冒烟用）。随机初始化时初始 loss 在**百量级**，
  加载预训练后应在**个位数**——看到三位数先怀疑权重没加载上。
- **SFT 的 `.pth` 只恢复权重，不恢复优化器状态**（有意设计）：存的是裸
  `model.state_dict()`（`scripts/sft.py` 的保存处），读回时也只 `load_state_dict`（`--checkpoint` 加载处）。
  要连优化器一起续训，用 `scripts/train.py --resume`（走 `Trainer.load_checkpoint`，含 optimizer / scheduler）。
- **权重加载不会打断 weight tying**：加载器只 `copy_` 进现有参数，不替换 `nn.Parameter` 对象，
  `out_head.weight is tok_emb.weight` 始终成立（与 HuggingFace `lm_head` / `wte` 的语义一致）。
- SFT 数据**不联网**：只能来自 `--json` 或 yaml 的 `data.local_json`，都没有就直接报错退出。

---

## 项目结构

```
configs/            7 份 yaml，三种形态：gpt2-{small,tiny,medium}（架构）、
                    train-{default,demo}（运行）、sft-*（嵌套，model.config 再指向架构 yaml）
src/my_llm/
  config.py         GPTConfig —— 架构配置的唯一来源
  model/            norm.py · attention.py · block.py · gpt.py（模型本体）
  weights/          hf.py（HuggingFace 权重）· openai_tf.py（OpenAI 原版 TF checkpoint）
  tokenizer/        protocol.py（Tokenizer 协议）· tiktoken_impl.py（tiktoken 实现）
  data/             dataset.py（滑窗数据集）· dataloader.py
  train/            config.py（TrainConfig）· losses.py · metrics.py（perplexity）·
                    scheduler.py（warmup+cosine）· trainer.py（续训/累积/混合精度）
  generate/         sampling.py（temperature/top-k/top-p）· kv_cache.py
  finetune/         sft.py（Alpaca 指令微调）
  utils/            seed.py · logging.py · viz.py
scripts/            train.py · generate.py · sft.py · download_weights.py
tests/              快测 12 个文件 + parity / crossload / kv_cache 三个慢测套件
notebooks/          01_tokenizer · 02_attention · 03_train_and_generate（教学向）
docs/               00-现状盘点.md（v1 全量审计，所有 `文件:行号` 证据的出处）
```

---

## 测试

```bash
make test        # 快测 83 个，~15s，不联网
pytest -m slow   # 慢测 22 个（parity 16 + crossload 2 + kv_cache 4），首次会下载权重
```

合计 **105 个用例**。

> ⚠️ **慢测必须显式加 `-m slow`**：`pyproject.toml` 里写着 `addopts = "-m 'not slow'"`，
> 所以 `pytest tests/test_parity_hf.py -v` 不加 `-m slow` 会**静默选 0 个用例**，
> 看起来"通过了"，其实什么都没跑。加 `-m slow` 才是真跑。

三个慢测套件各管一件事：

| 套件 | 管什么 |
|---|---|
| `test_parity_hf.py` | 与 HuggingFace 逐层对拍（logits / 12 层隐状态 / 注意力权重 / 嵌入层） |
| `test_crossload.py` | 两条权重加载路径互相对拍（HuggingFace vs OpenAI TF） |
| `test_kv_cache.py` | KV cache 与无 cache 路径逐 token 一致 + 耗时对比 |

> ⏱ 慢测里的 `test_tf_and_hf_agree` 要下载 OpenAI 原版 TF 检查点（475MB），
> 而那个源在国内只有几百 KB/s——干净环境下实测首次跑了约 20 分钟。
> 只想验证 parity 可以先跑 `pytest -m slow -k "not tf_and_hf_agree"`。

parity 失败时**先看 diff 量级再定位**：`1e-13` 量级是正常；`~1e-5` 通常是忘了 `eval()`；
`~1e-3` 是 GELU 变体不对或掩码用了 `-inf`；`1e-2~1e-1` 是漏了转置或 QKV 顺序错；
`>= 1e0` 是权重整体没加载上。

---

## 硬件要求

实测环境：RTX 5060 Laptop **8 GB**（WSL），fp32 + AdamW，随机初始化
（与加载预训练权重的显存占用一致，权重本身大小相同）。

| 配置 | batch × context | 峰值显存 | 结论 |
|---|---|---|---|
| gpt2-small（124M） | 2 × 1024 | **5.45 GiB** | ✅ 8 GB 卡可正常训练 |
| gpt2-medium（355M） | 1 × 1024 | **9.05 GiB** | ⚠️ 已超 8 GB 物理显存，WSL 下靠共享内存才跑完，会明显变慢；实用上要降 batch / context 或改用 bf16 |
| gpt2-large（774M） | — | 约 12 GiB（估算） | ❌ 8 GB 卡必然 OOM |
| gpt2-xl（1.5B） | — | 约 24 GiB（估算） | ❌ 8 GB 卡必然 OOM |

large / xl 两行是**估算**（`参数量 × 16 字节` = fp32 权重 + 梯度 + AdamW 两个动量），
还没算激活值——真实占用只会更高，所以结论不会翻转。

CPU 也能跑全部功能（快测与 parity 都强制 CPU + fp32），只是训练慢；
`make demo` 与 notebook 一律用 `configs/gpt2-tiny.yaml`。

---

## 与 v1 的关系

v2 是对 [v1（`my-LLM-from-scratch`）](https://github.com/tochikoma777/my-LLM-from-scratch)的**完全重写**，
不是打补丁：从扁平脚本（`from language_module import GPTModel`，绑死 CWD）改成 `src/my_llm/` 正式包，
配置从三处互相矛盾的 `GPT_CONFIG_124M` 字面量收敛到 `GPTConfig` + yaml，并补上了 v1 完全没有的
调度器、perplexity、KV cache、checkpoint 续训，以及这套数值对齐验证。

v1 的完整审计（每一项都带 `文件:行号` 证据）见 [`docs/00-现状盘点.md`](docs/00-现状盘点.md)；
v2 相对 v1 的逐条变化见 [`CHANGELOG.md`](CHANGELOG.md)。

顺便列三个 v1 里真实存在、v2 已修掉的坑——它们都不影响"生成出来像人话"，
所以靠肉眼永远发现不了：

- **weight tying 被打断**：v1 的 `assign()` 造出的是**副本**，加载后 `out_head` 与 `tok_emb`
  是两个独立张量，微调时各自更新而漂移。
- **掩码的值与时机**：`-torch.inf` → `torch.finfo(dtype).min`，且必须填在**缩放之后**的分数上。
- **`qkv_bias` 默认值**：v1 从零训练用 `False`，与 GPT-2 规格矛盾，也会让权重加载在
  给 `bias=None` 赋值时崩溃。
