# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Repository state

`v2-work/` 是 v2 重写的工作区；`../v1-reference/`（my-LLM-from-scratch，tochikoma777）是**只读参考**（见下方硬约束 3）。

截至 2026-10-03：HEAD 为 `2b32a3f feat(R3c)`（共 7 次提交，59 个被跟踪文件）。
已落地的是「工程地基 + 模型内核 + 配置层 + HF 权重加载 + OpenAI TF 权重加载 +
快测 / parity / crossload **三套件**」，其余模块仍是**类型完整的存根**
（填充顺序见下方「存根填充优先级（P0→P3）」）。

**已实现**（可运行）：
- `src/my_llm/config.py` — `GPTConfig` dataclass，唯一配置来源；`from_yaml` / `gpt2_small()` / `gpt2_tiny()`
- `src/my_llm/model/` — `norm.py`（LayerNorm + NewGELU）、`attention.py`（MultiHeadAttention）、
  `block.py`（FeedForward + TransformerBlock）、`gpt.py`（GPTModel，含 weight tying）
- `src/my_llm/tokenizer/` — `protocol.py`（Tokenizer 协议）、`tiktoken_impl.py`（`TiktokenTokenizer` +
  工厂 `build_tokenizer(name="gpt2")`）
- `src/my_llm/weights/hf.py` — HF state_dict → `GPTModel` 的完整键名映射：`load_hf_weights_into_gpt`
  （返回**未匹配键列表**，便于断言「没有漏映射」）、`load_weights_from_hf`（严格版，并校验 tie 仍成立）、
  `hf_key_to_ours`、`load_hf_state_dict`、`gpt_config_from_hf`。要点：`c_attn` 按 q→k→v 切分、
  Conv1D 形态权重逐层 `.T`、`lm_head.weight` 跳过（硬约束 1）
- `src/my_llm/weights/openai_tf.py` — OpenAI TensorFlow 检查点加载：`download_file`（幂等 + 超时重试）、
  `download_and_load_gpt2`、`load_gpt2_params_from_tf_ckpt`、`load_openai_tf_weights_into_gpt`
  （同样返回未匹配键列表）。tensorflow 惰性导入；`assign` 与 v1 不同——它返回张量而非新
  `nn.Parameter`，调用方 `copy_` 进现有参数（见模块 docstring 第 3 点）

**存根**（函数体 `raise NotImplementedError`，`src/` 下共 12 个文件）：
`data/dataset.py`、`data/dataloader.py`、`finetune/sft.py`、`generate/sampling.py`、`generate/kv_cache.py`、
`train/losses.py`、`train/metrics.py`、`train/scheduler.py`、`train/trainer.py`、`utils/seed.py`、
`utils/logging.py`、`utils/viz.py`
（`scripts/sft.py` 本体也一样会抛，不计入上面 12 个。）

`tests/` 是**三套件**：`conftest.py` + `test_norm.py` / `test_tying.py` / `test_attention.py` /
`test_tokenizer.py` / **`test_parity_hf.py`** / **`test_crossload.py`**。快测 15 个用例（`pytest -q` 全绿，~1s），
parity 慢测 16 个用例（`pytest -m slow -v` 实测 **16 passed，~25s**，首次运行会真实下载 `gpt2` 权重），
crossload 慢测 2 个用例（双路径交叉验证，实测 **2 passed，~40s**，见 Environment 里 TF 权重的坑）。
`notebooks/` 只有 `.gitkeep`。
`configs/` 有 5 份 yaml：`gpt2-small.yaml` / `gpt2-tiny.yaml` / `gpt2-medium.yaml`（架构），
以及 `sft-alpaca.yaml` / `sft-medium-bf16.yaml`（**嵌套**运行配置，见 Commands 里的坑）。
`docs/00-现状盘点.md` 是 v1 的完整审计报告（含行号证据、权重映射表、与 HF 的架构差异、取舍建议），
动手前优先读它，不要凭记忆重写结论。

## Environment

- Python 3.11.15 / torch 2.12.0+cu130 / transformers 5.12.0（本机实测）。**本机有 GPU**
  （RTX 5060 Laptop，8GB），但 parity 测试强制 CPU + fp32（硬约束 2），不要因为看到 CUDA 可用就改用 GPU。
- `transformers` 已装（仅 parity 用），pyproject 里锁 `>=5.12,<6`：parity 断言依赖 `GPTConfig()` 默认值与
  `NewGELUActivation` 实现，6.x 一改整套 parity 失效，不要放宽上界。
- `tensorflow` **未装**——它只是 `weights/openai_tf.py` 的可选依赖，不要把它加回主依赖。
- 网络：PyPI 走 tuna 镜像可通；**github.com 不可达**。因此
  - `pre-commit` 无法拉取新的 hook 版本（`rev` 改不动，改了会卡死在初始化）；
  - parity 测试下载 HF 权重依赖镜像源，不要删掉两处设置：`Makefile:3` 的
    `export HF_ENDPOINT ?= https://hf-mirror.com`（覆盖 `make` 目标），以及
    `scripts/download_weights.py:24` 的 `os.environ.setdefault(...)`（直接 `python scripts/...` 时的兜底）。
- HF 缓存不在默认位置：本机 `HF_HOME=/data/cache/huggingface`（不是 `~/.cache/huggingface`），
  排查「到底下没下权重」时别找错目录。`gpt2` 权重约 548MB，本机已缓存，
  所以 `pytest -m slow` 现在不用联网就能跑完。
- **`tensorflow-cpu` 已装在本机（2.21.0）**，只为 `weights/openai_tf.py` 这一条可选路径服务；
  `pyproject.toml` 主依赖里**不要**加 TF（CONTRIBUTING 硬规则 6）。TF 2.21 里
  `tf.train.list_variables` / `load_variable` 仍可用，不需要退回旧版。
- **OpenAI 的 Azure 源（`openaipublic.blob.core.windows.net`）在国内极慢**（实测）：
  单线程 ~45 KB/s（475MB 要 3 小时）；8 并发 Range ~348 KB/s；`aria2c -x16 -s16` 约 14 分钟（~590 KiB/s）。
  因此想跑 `tests/test_crossload.py::test_tf_and_hf_agree`，先用多线程工具把 7 个文件预置到
  `outputs/openai-tf/124M/`（该目录已被 `.gitignore` 覆盖），我们的 `download_file` 会因
  `Content-Length` 一致自动跳过（实测 7/7 skip）。
  两个坑：
  - **别中途 kill aria2**：它预分配文件，`ls` 显示的已是最终大小，段数据却还没写完。
    拿这种文件给 TF 读会报 `Checksum does not match: stored ... vs. calculated ...`——
    这不是 TF 侧的 bug，也不是我们的映射写错了，文件确实是坏的，删掉重下即可。
  - **下完用 Azure 的 `Content-MD5` 校验**：124M 的 `model.ckpt.data-00000-of-00001`
    正确值是 size `497759232` / md5 `f48b9cf1a525a603be52258f69bf9162`。

## Commands

```bash
make install     # pip install -e ".[dev,viz]" + pre-commit install
make lint        # ruff check src tests scripts + mypy src（当前全绿）
make fmt         # ruff format src tests scripts
make test        # pytest -q，跑 15 个快测（跳过 slow）
make test-full   # pytest -q -m slow -v，parity + crossload，共 18 个用例，首次会下载 GPT-2 权重
                 # （实测 18 passed / 15 deselected，约 41s；crossload 首次要下 OpenAI TF 权重，见 Environment）
make check       # lint + test
make demo        # tiny 配置训练 + 生成（⚠️ 见下方警告，当前跑不通）
make clean       # 清缓存

# scripts/ 下唯一能真跑通的入口（其余都是存根或有 TODO 占位实参）
python scripts/download_weights.py --source hf --model gpt2
python scripts/download_weights.py --source openai --model-size 124M
#   → OpenAI 权重默认落到 outputs/openai-tf/<model-size>/（硬约束：输出统一 outputs/）
#   → 脚本内部已 os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")，
#     所以不必手动加前缀；要换源时自己 export HF_ENDPOINT 覆盖即可

# 自查还剩多少存根（当前 12 个 src/ 文件 + scripts/sft.py）
grep -rl NotImplementedError --include="*.py" src scripts
```

> ⚠️ **提交前不要手动格式化**：pre-commit 会跑 ruff / ruff-format / mypy / detect-secrets。
> hook 改写文件后**重跑一次提交**即可，**第一次提交失败是正常的**，不要 `git commit --no-verify`。

> ⚠️ **`make demo` / `scripts/*.py` 目前无法端到端跑通**，缺口按优先级拆开看：
> - `scripts/generate.py:18` 直接 import `generate/sampling.py`（**P0**）；
> - `scripts/train.py:19` 直接 import `train/trainer.py`（**P1**），而 `Trainer.train()` 内部
>   要吃 `data/dataloader.py`（**P0**）产的 loader——现在两处都还是 `None` 占位
>   （`scripts/train.py:55,58`）。
> 所以 `make demo` 要等 **P0 + P1 都完成**才是真闭环，不是只补 P0。
> 实测 `python scripts/train.py --config configs/gpt2-tiny.yaml` 抛 `NotImplementedError`（`train/trainer.py:147`）。
> 另外 `scripts/train.py:55`（`optimizer=None`）与 `scripts/generate.py:54`（`idx=None`）是标了 `TODO` 的占位实参，
> 靠 `# type: ignore` 过 mypy，不是已接通的逻辑。
> 当前可运行的端到端路径只有两条：模型前向
> `GPTModel(GPTConfig.gpt2_tiny())(torch.zeros(2, 16, dtype=torch.long))` → `[2, 16, 50257]`，
> 以及「HF 权重加载 + parity」（`pytest -m slow -v`，见下）。

单个文件 / 单个用例：

```bash
pytest tests/test_tying.py -v                          # 单文件
pytest tests/test_tying.py::test_head_shares_storage -v  # 单用例
pytest tests/test_norm.py -k "gelu or norm" -v         # 按名字筛
pytest tests/test_tokenizer.py -k "roundtrip" -v       # 参数化用例按 id 筛
pytest -m slow -v                                      # 跑 parity；命令行 -m 覆盖 addopts 的 'not slow'
pytest "tests/test_parity_hf.py::test_block_hidden_states[3]" -m slow -v  # 单个 parity 用例（路径要加引号）
mypy src/my_llm/model/gpt.py                           # 单文件类型检查
ruff check src/my_llm/model                            # 局部 lint（不影响全局）
```

补充约定：
- pytest 默认 `addopts = "-m 'not slow'"`，即日常不跑联网测试；**任何会下载真实权重的测试必须标 `@pytest.mark.slow`**。
  注意 `pytest tests/test_parity_hf.py -v` **不加 `-m slow` 会静默选 0 个用例**（实测「16 deselected / 0 selected」），
  不是文件坏了，加 `-m slow` 即可。
- **parity 有专属坑**（都是踩出来的，改 `tests/test_parity_hf.py` 前先看这里）：
  - HF 侧必须 `attn_implementation="eager"`（`tests/conftest.py:80`）：默认的 sdpa 不返回注意力权重，
    `output_attentions=True` 时 `attentions` 是空元组；
  - transformers 5.x 会把 `hidden_states[-1]` 换成 `ln_f` 之后的结果，**不是**最后一个 block 的输出，
    直接拿 `output_hidden_states=True` 对拍会得到 1e0 量级的假失败；逐 block 比对必须挂 forward hook
    （`tests/test_parity_hf.py:103` `_hf_block_outputs`）；
  - 失败时**先看 diff 量级再定位**（`tests/test_parity_hf.py:19-23`）：

    | 量级 | 大概率原因 |
    |---|---|
    | `0 ~ 1e-6` | 正常 fp32 噪声 |
    | `~1e-5` | 忘记 `eval()`（dropout 没关） |
    | `~1e-3` | GELU 变体不对，或掩码用了 `-inf` |
    | `1e-2 ~ 1e-1` | 漏了转置，或 QKV 切分顺序错 |
    | `>= 1e0` | 权重整体没加载 |

- **crossload（`tests/test_crossload.py`）是两条权重路径的对拍**，与 parity 互补：
  `test_tf_and_hf_agree` 用真实 OpenAI TF 检查点（需 TF + 475MB 下载）；
  `test_openai_tf_loader_matches_hf_loader` 把 HF state_dict 摆成 TF 嵌套参数形态喂给 TF 加载器，
  **不需要 TF、不需要额外下载**就能验证 QKV 顺序 / 转置 / g-b 映射（实测 diff 精确为 0）。
  后者成立的前提是 TF 与 HF 的 Conv1D 排布一致（`c_attn` 都是 `(emb, 3*emb)`），
  所以 HF 权重可以直接充当 TF 参数。
  TF 未装或下载失败时必须是 **skip**（`pytest.importorskip` / `pytest.skip`），不能用 fail 掩盖。

- `tests/conftest.py` 提供 session 级 fixture：`tiny_cfg`（读 `configs/gpt2-tiny.yaml`）、`tiny_model`
  （`torch.manual_seed(0)` 初始化、`eval()`、CPU+fp32）、`sample_ids`（`(2,16)` 固定 token 批次），
  以及模块常量 `DEVICE="cpu"` / `DTYPE=torch.float32` / `REPO_ROOT` / `TINY_CONFIG_PATH` / `SMALL_CONFIG_PATH`
  （`REPO_ROOT` 用 `__file__` 定位，换 CWD 不崩）。
  新测试优先复用这些 fixture，不要自己 new 一份配置。
  另有三个 **slow 专用** fixture：`hf_gpt2`（HF 官方 `gpt2`，eager）、`our_gpt2_loaded`（装上同一份权重的
  `GPTModel`）、`parity_fp64`（两者的 fp64 深拷贝）——**它们会联网下载，只能被标 `@pytest.mark.slow` 的用例使用**。
  （`tokenizer` fixture 是 module 级，定义在 `tests/test_tokenizer.py:15`，不在 conftest 里。）
- 包通过 editable 安装导入（`import my_llm`）；脚本 `scripts/*.py` 从**仓库根目录**运行，
  `--config` 等路径是相对根目录的（如 `configs/gpt2-tiny.yaml`）。
- `pyproject.toml` 里 `[tool.hatch.build.targets.wheel] packages = ["src/my_llm"]` 是必需的：
  hatchling 按项目名推断的是 `src/my_llm_from_scratch/`，删掉会让 `pip install -e .` 失败。
- **yaml 有两种形态**，别喂错：`configs/gpt2-*.yaml` 是**扁平**架构配置（顶层键 = `GPTConfig` 字段），
  可直接 `GPTConfig.from_yaml(...)`；`configs/sft-*.yaml` 是**嵌套**运行配置，顶层是
  `model` / `data` / `train` / `generation` / `output`，其中 `model.config` 指向一份架构 yaml。
  把 sft yaml 直接喂给 `from_yaml` 会抛 `KeyError: 缺少必填字段 [...]`（实测如此），
  得先取 `model.config` 指向的路径再构造 `GPTConfig`。
- **导入层级**：只有根包 `src/my_llm/__init__.py` 不重导出（仅 `__version__`，`:25`），
  **子包都重导出了**：`from my_llm.model import GPTModel`、`from my_llm.tokenizer import build_tokenizer`
  均可；需要某个具体类时写深路径 `from my_llm.model.gpt import GPTModel` 也对，两者都行，
  但不要写 `from my_llm import GPTModel`（会 ImportError）。

## CI 与容器

`.github/workflows/ci.yml` 有两个 job，本地复现时注意差异：

| job | 触发 | Python | 步骤 |
|---|---|---|---|
| `test` | push / PR | 3.10 / 3.11 / 3.12 矩阵 | `ruff check src tests scripts` → `mypy src` → `pytest --cov=my_llm --cov-report=xml` |
| `parity` | **仅 push 到 main** | 3.11 | `pytest -m slow -v`，通过 job env 注入 `HF_ENDPOINT=https://hf-mirror.com` |

要在本地复刻 CI：用 `pip install -e ".[dev]"`（**不带** `viz`，CI 不装 seaborn）。
另有 `Dockerfile`（`python:3.11-slim`，`HF_ENDPOINT` 已写进镜像 ENV，CMD 为 `scripts/generate.py`）——
由于 `generate/sampling.py`、`train/trainer.py` 等仍是存根，它目前只能用来验证安装 / import / HF 权重加载路径。

## 文档地图

| 文件 | 看什么 |
|---|---|
| `README.md` | 第一屏 parity diff 表（v1 行为 vs HF vs v2 状态，含证据列）、项目结构、已知状态 |
| `CONTRIBUTING.md` | 6 条硬规则 + 搬运 v1 代码的规矩 + PR 要求 |
| `docs/00-现状盘点.md` | v1 全量审计，所有 `路径:行号` 证据的唯一出处 |
| `CHANGELOG.md` | 版本演进记录 |

`CONTRIBUTING.md` 里有本文件未重复收录、但同样必须遵守的几条：`src/` 必须通过
`mypy --strict`（`tests/` 可无注解）、输出统一落到 `outputs/` 且不用 CWD 相对路径、
改动 `model/` 必须同步 parity 测试、禁止 `git commit --no-verify`。

## Architecture

按层划分，越靠下越基础；当前只有 `config` + `model` + `tokenizer` + `weights/` 有真实实现，
其余包内的模块全是存根：

| 包 | 职责 | v1 来源 |
|---|---|---|
| `config.py` | `GPTConfig`（vocab_size / context_length / emb_dim / n_layers / n_heads / drop_rate / qkv_bias），`qkv_bias` 默认 True；`from_yaml` 只吃扁平 yaml | v1 三处冲突的 `GPT_CONFIG_124M` |
| `configs/` | `gpt2-{small,tiny,medium}.yaml` 扁平架构配置；`sft-*.yaml` 嵌套运行配置（`model.config` 再指向架构 yaml） | v1 的 `__main__` 字面量 |
| `model/` | 模型本体。`norm.py` / `attention.py` / `block.py` / `gpt.py` | `language_module.py` |
| `weights/` | `hf.py` 与 `openai_tf.py` 均已实现：前者是 parity 取数入口，后者是 TF→PT 兼容层（v1 独特资产），两条路径的数值等价性由 `tests/test_crossload.py` 交叉验证 | `module_load_param.py` |
| `tokenizer/` | `protocol.py`（`Tokenizer` runtime_checkable 协议）+ `tiktoken_impl.py`（`TiktokenTokenizer`、`build_tokenizer()`）（`bpe.py` 留作 P3） | 四处重复的 `tiktoken.get_encoding("gpt2")` |
| `data/` | 滑窗数据集 / dataloader | `data_preprocess.py` |
| `train/` | `losses.py` / `metrics.py`(perplexity) / `scheduler.py`(warmup+cosine) / `trainer.py`(续训+累积+AMP) | `module_train.py` |
| `generate/` | `sampling.py`(greedy/temp/top-k/top-p) / `kv_cache.py` | `module_load_param.py:generate`、`generate_text_simple.py` |
| `finetune/` | Alpaca SFT | `module_fine_tuning.py` |
| `utils/` | `seed.py` / `logging.py` / `viz.py` | v1 无对应 |

**`weights/` 是并列的两条 loader**，共享同一套键名约定（QKV 切分顺序、Conv1D 转置、g-b 后缀）：
`hf.py` 是主力（parity 取数入口），`openai_tf.py` 是 v1 资产（TF→PT 兼容层）。
**任一边改键名映射，都必须同步 `tests/test_crossload.py`**——CONTRIBUTING 硬规则 4 只写了
「改 `model/` 必须同步 parity」，没覆盖 `weights/`，这条在这里补齐。

## 存根填充优先级（P0→P3）

12 个 `src/` 存根（外加 `scripts/sft.py`）按「完成后能做什么」划分，每个优先级是一个可验收的里程碑：

| 优先级 | 判据（完成后能做什么） | 模块 |
|---|---|---|
| **P0** | 能训练 + 能生成，demo 最小闭环 | `data/dataset.py`、`data/dataloader.py`、`train/losses.py`、`train/scheduler.py`、`train/metrics.py`、`generate/sampling.py` |
| **P1** | CLI 可跑 + 有加速数据 | `train/trainer.py`（checkpoint 续训 / 梯度累积 / 混合精度）、`generate/kv_cache.py`、`scripts/train.py`、`scripts/generate.py` |
| **P2** | Alpaca 微调可跑 + 结果可复现 | `finetune/sft.py`、`scripts/sft.py`、`utils/seed.py`、`utils/logging.py` |
| **P3** | 门面与传播物料 | `utils/viz.py`、`notebooks/`、顶层 API 导出（`from my_llm import GPTModel`）、自实现 BPE |

注意 `scripts/train.py` / `scripts/generate.py` 虽然在 P1，但它们当前是**已存在的占位脚本**
（`optimizer=None` / `idx=None` 两个 TODO 实参），不是待新建文件——P1 的工作包含把它们接通。

模型侧已确认的不变量（改动后必须仍然成立）：
- `model/gpt.py:65` `self.out_head.weight = self.tok_emb.weight` —— 同一 `nn.Parameter` 对象；
- `model/attention.py:120` 掩码用 `torch.finfo(dtype).min`，且填在**缩放之后**的分数上（`:113` 缩放，
  `:118` 用 `.to(torch.bool)`/`cast` 绕开 mypy strict 对 `.bool()` 的报错）；
- `config.py:48` `qkv_bias: bool = True`。

## Conventions and gotchas

- **中文优先**：注释、docstring、日志与参数帮助文字均用中文，标识符用 ASCII。与既有代码保持一致。
- **配置只能来自 yaml**：超参进 `GPTConfig` 或 `configs/*.yaml`，不要写进 `__main__`（硬约束 4）。
- **ruff 版本分歧**：本地 ruff 0.16.9，pre-commit 里锁的是 v0.6.9（`.pre-commit-config.yaml:3`），
  两者对多行 `assert` 的换行风格结论相反，会互相改写文件（已踩过一次，导致提交反复失败）。
  规避办法：断言消息先赋变量、写成单行；**不要**去改 hook 的 `rev`（GitHub 不可达，改了装不上）。
- **mypy 同样有版本分歧**：本地 mypy 2.3.1，pre-commit 锁的是 v1.11.2（`.pre-commit-config.yaml:9`）。
  结论冲突时以 **hook（v1.11.2）为准**，同样不要改 `rev`。
- **pre-commit 的 mypy 跑在隔离环境**：`additional_dependencies` 必须显式列出 `torch, numpy, types-PyYAML`，
  否则 `config.py` 的 `import yaml` 会报 `import-untyped`（本地能过是因为本地装了 pyyaml 本体）。
- **产物不要入库**。注意 `.gitignore` **一行只能写一个模式**：写在一行上（如过去的
  `outputs/ runs/`）会被 git 当成一个带空格的模式、完全不生效。2026-10-02 修的就是这处——
  `outputs/` 原先其实没被忽略，OpenAI TF 权重的 475MB 差点混进提交。
  2026-10-02 已把全部同类问题一次拆行修好，并用 `git check-ignore -v` 逐条验证生效
  （目录型模式要拿真实目录测，对不存在的路径 `git check-ignore` 会报 NOT IGNORED，属误报）。
- 提交前无需手动格式化（ruff / ruff-format / mypy / detect-secrets 都由 hook 跑），
  细节见 Commands 段的 ⚠️ 提示。

## 项目硬约束（v2 重构）

1. **weight tying**：任何权重加载代码都不得给 `out_head` 赋值，**也不得替换 `tok_emb.weight`
   这个 `nn.Parameter` 对象**——后者是"间接打断"，比前者更隐蔽，后果一样。
   weight tying 由 `GPTModel.__init__` 保证（`self.out_head.weight = self.tok_emb.weight` 是**同一个对象**）；
   一旦把 `tok_emb.weight` 换成新对象，`out_head.weight` 仍指向旧的那块，tie 当场断开。
   v1 的 `assign`（`LOAD:235`）返回的正是新 `nn.Parameter`，所以即便不碰 `out_head` 也会中招。
   因此加载器**只准 `copy_` 进现有参数**：`weights/hf.py` 与 `weights/openai_tf.py` 都这么做，
   且两者结束处都有 `assert gpt.out_head.weight is gpt.tok_emb.weight` 兜底。

2. **parity 测试规格**：强制 `device="cpu"`、`dtype=torch.float32`，且是**双断言**（`tests/test_parity_hf.py:49` `_assert_parity`）：
   - **fp64 绝对 `1e-5`**：两边都转 fp64 再比，实测互差 ~3e-13，是「数学正确性」的硬证据，阈值不可动。
   - **fp32 `1e-5 × max(1, max|ref|)`**：保留真实推理 dtype，按参考张量量级缩放（等价相对 1e-5）。
     不能改成 fp32 绝对 1e-5：GPT-2 残差流量级到 3e3，fp32 自噪声就有 1e-4，**低于噪声地板的阈值只会产出
     「有时过有时不过」的幽灵失败**。
   - 禁止再放宽任何一侧。原因：GPU 上 fp32 矩阵乘法有非确定性，噪声与阈值同量级时结果不可复现。

3. **v1 只读**：`../v1-reference` 是只读参考，任何情况下不得修改其中的文件。

4. **配置唯一来源**：所有超参必须来自 `configs/*.yaml`。
   禁止在 `if __name__ == "__main__"` 里硬编码配置——这是 v1 最大的结构性缺陷，不得重现。
