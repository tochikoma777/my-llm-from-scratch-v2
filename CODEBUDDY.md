# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Repository state

`v2-work/` 是 v2 重写的工作区；`../v1-reference/`（my-LLM-from-scratch，tochikoma777）是**只读参考**（见下方硬约束 3）。

截至 2026-10-05：HEAD 为 `5b3ddc4 feat(R4c): SFT 默认加载预训练权重`（共 22 次提交；
父提交 `8b558b9` 才是 SFT 流水线本体，本提交只加了权重三态 `--checkpoint` / `--no-pretrained`），
**`src/` 存根已归零**（P0→P2 全部落地；`scripts/sft.py` 也已接通，见下方「存根填充优先级」）。
已落地的是「工程地基 + 模型内核 + 配置层（架构 + 运行两份）+ HF/OpenAI 权重加载 +
数据层 + 训练工具层 + Trainer + 生成层（采样 + KV cache）+ **微调层（Alpaca SFT）+ utils 三件套**
+ 三个 CLI + **三个教学 notebook**」。测试是快测 / parity / crossload / kv_cache 四套件
（实测 `83 passed, 22 deselected`；`pytest -m slow` 实测 22 passed）。
剩余未做的只剩 P3 的顶层 API 导出与自实现 BPE（`tokenizer/bpe.py`）。

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
- `src/my_llm/train/config.py` — `TrainConfig`（+ `OptimizerConfig` / `SchedulerConfig` / `DataConfig`），
  训练运行配置层，与 `GPTConfig` 并列，从 `configs/train-*.yaml` 读
- `src/my_llm/data/` — `dataset.py`（`GPTDatasetV1` 滑窗数据集）、`dataloader.py`（`create_dataloader_v1`）
- `src/my_llm/train/` — `losses.py`（`calc_loss_batch` / `calc_loss_loader`）、`metrics.py`
  （`loss_to_perplexity` / `evaluate_perplexity` / `evaluate_model`）、`scheduler.py`
  （`warmup_cosine_lr` / `get_cosine_schedule_with_warmup`）、`trainer.py`
  （`Trainer` + `TrainerConfig` + `save_checkpoint` / `load_checkpoint`，支持 checkpoint 续训、
  梯度累积、fp16/bf16/fp32）
- `src/my_llm/generate/` — `sampling.py`（`apply_temperature` / `apply_top_k` / `apply_top_p` /
  `sample_next_token` / `generate`）、`kv_cache.py`（`KVCache` + `generate_with_cache`，
  不碰 `model/`，见 Architecture 节）
- `src/my_llm/finetune/sft.py` — Alpaca 指令微调：`InstructionDataset`（构造时一次性分词）、
  `format_input`（**模板逐字照搬 v1 `SFT:270-277`，训练/推理共用**）、`custom_collate_fn`
  （变长补齐 + 损失屏蔽：`targets` 里**第一个 pad 保留为预测目标**，其余 padding 置 `ignore_index`）、
  `run_sft`（AdamW 微调循环，基线评估 + 每轮评估）、`generate_responses`（贪婪解码逐条生成，
  只解码新 token 再去掉 `### Response:` 前缀）。`__init__.py` 只导出前三个
- `src/my_llm/utils/` — `seed.py`（`set_seed` 播 `random`/`numpy`/`torch`/`torch.cuda`，
  `deterministic=True` 才开确定性算法；`get_generator` 不污染全局 RNG；`seed_worker` 作
  `worker_init_fn`）、`logging.py`（`get_logger` 取 `my_llm.*`；`configure_logging` **只由入口脚本调用**）、
  `viz.py`（`plot_losses` 双 x 轴 + `plot_attention_heatmap`，**只用 matplotlib 不用 seaborn**——
  seaborn 仅在 `viz` extra，CI 走 `.[dev]` 没有它）

**存根**：已归零。上一次还有的 4 个 `src/` 文件（`finetune/sft.py`、`utils/{seed,logging,viz}.py`）
与 `scripts/sft.py` 都在 `8b558b9`（R4c）落地。自查命令仍是
`grep -rl NotImplementedError --include="*.py" src scripts`（现在应无输出）。

`tests/` 是**四套件**：`conftest.py` + `test_norm.py` / `test_tying.py` / `test_attention.py` /
`test_tokenizer.py` / `test_data.py` / `test_metrics.py` / `test_scheduler.py` / `test_trainer.py` /
`test_sampling.py` / `test_sft.py` / `test_seed.py` / `test_viz.py` / **`test_parity_hf.py`** /
**`test_crossload.py`** / **`test_kv_cache.py`**。
快测 83 个用例（`pytest -q` 全绿，实测 **83 passed / 22 deselected，~8s**；
比 R4b 的 61 多了 `test_sft` 11 + `test_seed` 6 + `test_viz` 5），
parity 慢测 16 个用例（实测 **16 passed，~25s**，首次运行会真实下载 `gpt2` 权重），
crossload 慢测 2 个用例（双路径交叉验证，实测 **2 passed，~40s**，见 Environment 里 TF 权重的坑），
kv_cache 慢测 4 个用例（实测 **4 passed**，`pytest -m slow` 总计 **22 passed**）。
`notebooks/` 有 3 个教学 notebook（`01_tokenizer` / `02_attention` / `03_train_and_generate`），
由 `pyproject.toml` 的 `notebooks` extra 提供执行依赖，**CI 不跑**（见 Commands 段）。
`configs/` 有 7 份 yaml，三种形态：`gpt2-{small,tiny,medium}.yaml`（扁平架构）、
`train-{default,demo}.yaml`（扁平运行，对应 `my_llm/train/config.py:TrainConfig`）、
`sft-alpaca.yaml` / `sft-medium-bf16.yaml`（**嵌套**运行配置，见 Commands 里的坑）。
`docs/00-现状盘点.md` 是 v1 的完整审计报告（含行号证据、权重映射表、与 HF 的架构差异、取舍建议），
动手前优先读它，不要凭记忆重写结论。

## Environment

- Python 3.11.15 / torch 2.12.0+cu130 / transformers 5.12.0（本机实测）。**本机有 GPU**
  （RTX 5060 Laptop，8GB），但 parity 测试强制 CPU + fp32（硬约束 2），不要因为看到 CUDA 可用就改用 GPU。
- `transformers` 已装（仅 parity 用），pyproject 里锁 `>=5.12,<6`：parity 断言依赖 `GPTConfig()` 默认值与
  `NewGELUActivation` 实现，6.x 一改整套 parity 失效，不要放宽上界。
- `tensorflow` **不在主依赖里**——它只是 `weights/openai_tf.py` 的可选依赖，不要把它加回
  `pyproject.toml`（本机装了 `tensorflow-cpu` 只是为了跑 crossload，见下方）。
- 网络：PyPI 走 tuna 镜像可通；**github.com 可能不可达**。因此
  parity / crossload 下载 HF 权重依赖镜像源，不要删掉两处设置：`Makefile:3` 的
  `export HF_ENDPOINT ?= https://hf-mirror.com`（覆盖 `make` 目标），以及
  `scripts/download_weights.py:24` 的 `os.environ.setdefault(...)`（直接 `python scripts/...` 时的兜底）。
  - 实测：2026-10-04 `gh` CLI（api.github.com）与 `git push`（github.com HTTPS）均可用，
    首次推送成功。但不保证稳定，失败时按上面的策略处理。
  - `pre-commit` 已全量改成本地模式（见 Conventions），**不再联网拉 hook**，所以 CI/GitHub
    不可达不影响提交。
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
make test        # pytest -q，跑 83 个快测（跳过 slow，实测 83 passed / 22 deselected，~8s）
make test-full   # pytest -q -m slow -v，parity 16 + crossload 2 + kv_cache 4 = 22 个用例
                 # （实测 22 passed / 83 deselected；首次会下载 GPT-2 权重，见 Environment）
make check       # lint + test
make demo        # tiny 配置训练 1 轮（train-demo.yaml）+ 用 last.pt 生成（实测 ~12s，已跑通）
make clean       # 清缓存

# 三个端到端 CLI（generate / sft 不给 --checkpoint 时会拉 HF gpt2 权重）
python scripts/generate.py --config configs/gpt2-small.yaml --prompt "Every effort moves you"
python scripts/train.py --config configs/gpt2-tiny.yaml --train-config configs/train-demo.yaml
# SFT：数据必须来自 --json 或 yaml 的 data.local_json（离线策略：不联网下载数据）
python scripts/sft.py --json data/raw/instruction-sample.json --test-mode
python scripts/sft.py --json data/raw/instruction-sample.json --checkpoint outputs/model-sft.pth
python scripts/sft.py --json data/raw/instruction-sample.json --no-pretrained  # 随机初始化，仅冒烟

# 权重下载（唯一会联网的脚本）
python scripts/download_weights.py --source hf --model gpt2
python scripts/download_weights.py --source openai --model-size 124M
#   → OpenAI 权重默认落到 outputs/openai-tf/<model-size>/（硬约束：输出统一 outputs/）
#   → 脚本内部已 os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")，
#     所以不必手动加前缀；要换源时自己 export HF_ENDPOINT 覆盖即可

# 自查还剩多少存根（R4c 之后应无输出）
grep -rl NotImplementedError --include="*.py" src scripts
```

### 本地数据与产出目录（全新 clone 必读）

`data/raw/` 被 `.gitignore:18` 覆盖，**仓库里没有任何数据文件**（实测 `git ls-files data` 为空）。因此：
- `make demo` / `scripts/train.py` 会先撞上预检并抛 `FileNotFoundError: 语料不存在`
  （`scripts/train.py:89-90`，是明确报错不是崩溃）——需要自备 `data/raw/the-verdict.txt`；
- `scripts/sft.py --json ...` 同理，需要自备 `data/raw/instruction-sample.json`
  （Alpaca 三字段 `instruction` / `input` / `output`，样例见 `tests/test_sft.py:31-33`）；
- **快测不依赖任何数据文件**：`tests/` 里没有用例引用 `data/raw`（实测 83 passed），
  所以"没数据"只影响 CLI 与 `make demo`，不影响 `make test`。

产出统一落 `outputs/`（同样已 gitignore），当前实际布局：

| 路径 | 谁写的 / 谁读 |
|---|---|
| `outputs/checkpoints/last.pt` | `make demo` 训练写出，再由 `Makefile:27` 读回做生成 |
| `outputs/model-sft.pth` | SFT 默认 checkpoint 落点 |
| `outputs/instruction-data-with-response.json` | `run_sft` 结束后带模型回复的数据 |
| `outputs/openai-tf/<model-size>/` | OpenAI TF 权重（下载坑见 Environment） |

> ⚠️ **提交前不要手动格式化**：pre-commit 会跑 ruff check --fix / ruff format / mypy（`language: system`，
> 用的是本机已装的 ruff / mypy）。hook 改写文件后**重跑一次提交**即可，
> **第一次提交失败是正常的**，不要 `git commit --no-verify`。
> 注意 hook 里**没有** detect-secrets（本机未安装，见 `.pre-commit-config.yaml:31` 注释），
> 别凭 CONTRIBUTING.md 里的旧描述以为它还在。

> ℹ️ **`make demo` 已跑通**（`scripts/train.py` + `scripts/generate.py` 都已接通）。
> 两条 CLI 的形态：
> - `scripts/train.py --config <架构yaml> [--train-config <运行yaml>] [--epochs/--batch-size/--lr/--seed 覆盖] [--resume ckpt]`
>   训练超参**只能**来自 `configs/train-*.yaml`；`--train-config` 默认 `configs/train-default.yaml`
>   （`scripts/train.py:42`），`make demo` 用的是 `configs/train-demo.yaml`（1 epoch）；
>   `--data` 默认 `data/raw/the-verdict.txt`（`:48`，文件不在仓库里，见上一节）；
>   四个覆盖 flag 默认 `None`（`:51-54`），`None` 时回落 yaml；`--resume` 接 `Trainer` 的
>   checkpoint 续训（`:55`）；
> - `scripts/generate.py --config <架构yaml> --prompt "..."`；不给 `--checkpoint` 时拉 HF `gpt2` 权重。
>   其余 flag 全在 `scripts/generate.py:40-48`：`--max-new-tokens`（默认 50）、`--temperature`
>   （默认 0.0，`<=0` 走贪婪解码）、`--top-k`（默认 None）、`--top-p`（默认 None）、`--seed`（默认 123）；
> - `scripts/sft.py --sft-config <嵌套运行yaml> [--config <架构yaml>] --json <本地指令数据>`
>   **`--data` 已改名 `--sft-config`**（原名误导：它指向 yaml，真正的数据走 `--json`）。
>   架构 yaml 默认 `None` → 回落 `--sft-config` 里的 `model.config`（嵌套形态，
>   不能直接喂 `GPTConfig.from_yaml`，会 KeyError）；`--epochs/--lr/--seed` 默认 `None`，
>   回落 `train:` 段（`configs/sft-*.yaml` 里已补 `seed: 123`，否则没有兜底值会违反硬约束 4）。
>   权重三态：**缺省加载 HF `gpt2`** → `--checkpoint <本地 .pth>` → `--no-pretrained` 才随机初始化。
>   随机初始化时初始 loss 在**百量级**（124M 约 440），加载预训练后应在**个位数**
>   （实测 4.6 → 2.6），看到三位数先怀疑权重没加载上。
>
> 可运行的端到端路径：模型前向
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

# 本地覆盖率（CI 用的是 --cov-report=xml，见 .github/workflows/ci.yml:17）
pytest --cov=my_llm --cov-report=term-missing

# notebook 依赖（notebooks extra，主依赖不含）：jupyter / nbclient / nbformat / ipykernel
pip install -e ".[notebooks]"
```

> ℹ️ **notebook 是人工校验，CI 不跑**：三个 notebook 都会真实训练/画图，执行慢且
> 依赖绘图后端，**发版前需人工从零跑一遍**（重启内核顺序执行）确认无报错。
> 它们刻意不依赖 `data/raw/`——`03` 的语料写在 notebook 里，所以空目录也能跑通。

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
- **yaml 有三种形态**，别喂错：
  - `configs/gpt2-*.yaml` —— **扁平架构配置**（顶层键 = `GPTConfig` 字段），可直接 `GPTConfig.from_yaml(...)`；
  - `configs/train-*.yaml` —— **扁平运行配置**（顶层标量 + `optimizer` / `scheduler` / `data` 三个二级段），
    对应 `my_llm/train/config.py` 的 `TrainConfig`，一次 `yaml.safe_load` 解析完；当前有
    `train-default.yaml`（默认，10 epoch）与 `train-demo.yaml`（`make demo` 用，1 epoch）；
  - `configs/sft-*.yaml` —— **嵌套运行配置**，顶层是 `model` / `data` / `train` / `generation` / `output`，
    其中 `model.config` 再指向一份架构 yaml 的路径。
  把 sft yaml 直接喂给 `GPTConfig.from_yaml` 会抛 `KeyError: 缺少必填字段 [...]`（实测如此），
  得先取 `model.config` 指向的路径再构造 `GPTConfig`。
  反过来，train yaml **不是**给 `GPTConfig` 用的，别把两份搞混——架构与训练超参是分离的两份文件。
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
端到端可用：`scripts/generate.py` 已接通（采样 + KV cache），镜像能真正跑生成；
`finetune/` 与 `utils/*` 也已实现，`scripts/sft.py` 不再抛 `NotImplementedError`。

## 文档地图

| 文件 | 看什么 |
|---|---|
| `README.md` | 第一屏 parity diff 表（v1 行为 vs HF vs v2 状态，含证据列）、项目结构、已知状态 |
| `CONTRIBUTING.md` | 6 条硬规则 + 搬运 v1 代码的规矩 + PR 要求 |
| `docs/00-现状盘点.md` | v1 全量审计，所有 `路径:行号` 证据的唯一出处 |
| `CHANGELOG.md` | v2.0.0 相对 v1 的 Added / Changed / Fixed / Removed |
| `notebooks/*.ipynb` | 三个教学 notebook（分词器 / 注意力 / 训练+生成） |

> ℹ️ **README 与 CHANGELOG 已在 R5 重写（2026-10-06）**，用例数是「快测 83 + 慢测 22 = 105」。
> 细节（行号证据、硬约束、坑）仍以本文件为准——README 是给使用者看的，本文件是给干活的人看的。

`CONTRIBUTING.md` 里有本文件未重复收录、但同样必须遵守的几条：`src/` 必须通过
`mypy --strict`（`tests/` 可无注解）、输出统一落到 `outputs/` 且不用 CWD 相对路径、
改动 `model/` 必须同步 parity 测试、禁止 `git commit --no-verify`。

## Architecture

按层划分，越靠下越基础。**配置层是并列的两份**（架构 vs 运行），这是理解 CLI 的关键。
`src/` 下已无存根（R4c 完成），P3 只剩顶层 API 导出与自实现 BPE（`notebooks/` 已在 R5 补齐）：

| 包 | 职责 | v1 来源 |
|---|---|---|
| `config.py` | **架构**配置 `GPTConfig`（vocab_size / context_length / emb_dim / n_layers / n_heads / drop_rate / qkv_bias），`qkv_bias` 默认 True；`from_yaml` 只吃扁平 yaml | v1 三处冲突的 `GPT_CONFIG_124M` |
| `train/config.py` | **运行**配置 `TrainConfig`（`seed` / `num_epochs` / `batch_size` / `eval_freq` / `save_every` / `grad_accum_steps` / `grad_clip` / `precision` + `OptimizerConfig` / `SchedulerConfig` / `DataConfig` 三个二级 dataclass），从 `configs/train-*.yaml` 读。v1 把这些埋在 `__main__` 里 | v1 `TRAIN:541-546` 的 `OTHER_SETTINGS` |
| `configs/` | 共 7 份，三种形态：`gpt2-{small,tiny,medium}.yaml`（扁平架构）、`train-{default,demo}.yaml`（扁平运行）、`sft-*.yaml`（嵌套，`model.config` 再指向架构 yaml） | v1 的 `__main__` 字面量 |
| `model/` | 模型本体。`norm.py` / `attention.py` / `block.py` / `gpt.py` | `language_module.py` |
| `weights/` | `hf.py` 与 `openai_tf.py` 均已实现：前者是 parity 取数入口，后者是 TF→PT 兼容层（v1 独特资产），两条路径的数值等价性由 `tests/test_crossload.py` 交叉验证 | `module_load_param.py` |
| `tokenizer/` | `protocol.py`（`Tokenizer` runtime_checkable 协议）+ `tiktoken_impl.py`（`TiktokenTokenizer`、`build_tokenizer()`）（`bpe.py` 留作 P3） | 四处重复的 `tiktoken.get_encoding("gpt2")` |
| `data/` | 滑窗数据集 / dataloader | `data_preprocess.py` |
| `train/` | `config.py`(TrainConfig) / `losses.py` / `metrics.py`(perplexity) / `scheduler.py`(warmup+cosine) / `trainer.py`(checkpoint 续训 + 梯度累积 + fp16/bf16/fp32) | `module_train.py` |
| `generate/` | `sampling.py`(greedy/temp/top-k/top-p) / `kv_cache.py` | `module_load_param.py:generate`、`generate_text_simple.py` |
| `finetune/` | Alpaca SFT（`sft.py`）。`custom_collate_fn` 的 `ignore_index` 默认 `-100` 与 `train/losses.py` 的 `F.cross_entropy(mean)` 是同一套约定，**两边不能各说一套** | `module_fine_tuning.py` |
| `utils/` | `seed.py`（可复现）/ `logging.py`（`my_llm.*` 命名空间）/ `viz.py`（**只用 matplotlib，不引 seaborn**：seaborn 仅在 viz extra，CI 装的是 `.[dev]`） | v1 无对应 |

**`generate/kv_cache.py` 刻意不改 `model/`**：KV cache 需要"只算新 token、复用历史 K/V"，
而 `model/attention.py` 的 `forward(x)` 只吃一个参数、没有 cache 接口。给 `model/` 加
`use_cache` 会触发 CONTRIBUTING 硬规则 4（改 `model/` 必须同步 parity），风险远大于收益。
现在的做法是**实例级旁路**：预填充时用 `W_key` / `W_value` 的 forward hook 捕获 K/V
（拿到的是模块真实输出，不重算），解码时把每层 `att.forward` 换成 cache 版、
把 `pos_emb` 换成"当前绝对位置那一段"的 hook，`finally` 里 `del att.forward` 还原。
实测两条路径输出**逐 token 完全一致**，CPU 上加速 ~2.4x（数字见 README）。

**`weights/` 是并列的两条 loader**，共享同一套键名约定（QKV 切分顺序、Conv1D 转置、g-b 后缀）：
`hf.py` 是主力（parity 取数入口），`openai_tf.py` 是 v1 资产（TF→PT 兼容层）。
**任一边改键名映射，都必须同步 `tests/test_crossload.py`**——CONTRIBUTING 硬规则 4 只写了
「改 `model/` 必须同步 parity」，没覆盖 `weights/`，这条在这里补齐。

**改动 → 同步测试**的完整映射（硬规则 4 只覆盖 `model/`，其余在这里补齐）：

| 动了什么 | 必须同步什么 |
|---|---|
| `model/*`（任何影响数值的改动） | `tests/test_parity_hf.py`（CONTRIBUTING 硬规则 4） |
| `weights/hf.py` / `weights/openai_tf.py` 的键名映射 | `tests/test_crossload.py`（见上一段） |
| `finetune/sft.py`（尤其 `format_input` 模板、`custom_collate_fn` 的 `ignore_index`） | `tests/test_sft.py`，特别是 `test_ignore_index_matches_cross_entropy_semantics` |
| `generate/kv_cache.py` | `tests/test_kv_cache.py`：`test_cache_matches_plain_generation` 守「与无 cache 路径逐 token 一致」 |
| `config.py` / `train/config.py` 加字段 | 同步 `configs/*.yaml` 与对应 `from_yaml`；新字段没有 yaml 兜底值会直接 KeyError（违反硬约束 4） |

## 存根填充优先级（P0→P3）

原本 12 个 `src/` 存根（外加 `scripts/sft.py`）按「完成后能做什么」划分，每个优先级是一个可验收的里程碑。
**P0 / P1 / P2 已完成（`src/` 存根归零）**，只剩 P3：

| 优先级 | 判据（完成后能做什么） | 模块 | 状态 |
|---|---|---|---|
| **P0** | 能训练 + 能生成，demo 最小闭环 | `data/dataset.py`、`data/dataloader.py`、`train/losses.py`、`train/scheduler.py`、`train/metrics.py`、`generate/sampling.py` | ✅ 已完成 |
| **P1** | CLI 可跑 + 有加速数据 | `train/trainer.py`（checkpoint 续训 / 梯度累积 / 混合精度）、`generate/kv_cache.py`、`scripts/train.py`、`scripts/generate.py` | ✅ 已完成（`make demo` 实测跑通） |
| **P2** | Alpaca 微调可跑 + 结果可复现 | `finetune/sft.py`、`scripts/sft.py`、`utils/seed.py`、`utils/logging.py`、`utils/viz.py` | ✅ 已完成（`8b558b9`，SFT 默认加载 gpt2 预训练权重，实测 loss 4.6 → 2.6） |
| **P3** | 门面与传播物料 | 顶层 API 导出（`from my_llm import GPTModel`）、自实现 BPE（`tokenizer/bpe.py`） | ⏳ 未开始 |
| **P3** | 门面与传播物料 | `notebooks/01_tokenizer` / `02_attention` / `03_train_and_generate`（教学向，已逐个跑通） | ✅ 已完成（R5） |
| **P3** | 文档与现状同步 | **重写 `README.md` 与 `CHANGELOG.md`** | ✅ 已完成（R5） |

注意 `scripts/train.py` / `scripts/generate.py` 在 P1，且它们原本就是**已存在的占位脚本**
（`optimizer=None` / `idx=None` 两个 TODO 实参），P1 的工作包含把它们接通。

模型侧已确认的不变量（改动后必须仍然成立）：
- `model/gpt.py:65` `self.out_head.weight = self.tok_emb.weight` —— 同一 `nn.Parameter` 对象；
- `model/attention.py:120` 掩码用 `torch.finfo(dtype).min`，且填在**缩放之后**的分数上（`:113` 缩放，
  `:118` 用 `.to(torch.bool)`/`cast` 绕开 mypy strict 对 `.bool()` 的报错）；
- `config.py:48` `qkv_bias: bool = True`。

## Conventions and gotchas

- **中文优先**：注释、docstring、日志与参数帮助文字均用中文，标识符用 ASCII。与既有代码保持一致。
- **动 `finetune/sft.py` 前先读它的模块 docstring**：那里写了三条必须保留的语义——
  ① `format_input` 的三段式模板（v1 `SFT:270-277`）**逐字保持**，训练与推理共用同一个字符串；
  ② `custom_collate_fn` 的 `targets` 里**第一个 pad 保留为预测目标**、其余 padding 才置 `ignore_index`
  （丢了这条模型学不会产出 `<|endoftext|>`，会表现为"生成停不下来"，极易被误判成超参问题）；
  ③ collate 不带 `device` 参数、张量恒在 CPU（修 v1 `SFT:371-375` vs `:421` 的 device 不一致）。
  另外 `ignore_index` 默认 `-100` 与 `train/losses.py:41` 的 `F.cross_entropy(mean)` 是绑定的，
  由 `tests/test_sft.py::test_ignore_index_matches_cross_entropy_semantics` 守着，两边不能各说一套。
- **配置只能来自 yaml**：架构超参进 `GPTConfig` + `configs/gpt2-*.yaml`，训练超参进 `TrainConfig` +
  `configs/train-*.yaml`，不要写进 `__main__`（硬约束 4）。
- **`pre-commit` 已全量改为 `repo: local` + `language: system`**（`.pre-commit-config.yaml:13`，
  改于 HEAD `859bda2`）：hook 直接调用本机已装的 ruff 0.16.9 / mypy 2.3.1，不再拉远程 hook 环境。
  历史背景——原来 ruff 锁 v0.6.9、本地 0.16.9，两者对多行 `assert` 换行风格结论相反，
  导致「hook 改文件 → 重跑提交 → 本地又改回去」的死循环。现在不存在版本分歧了：
  **hook 结果与 `make lint` 完全一致**，不必再以 hook 版本为准。
- 由此带来两点：
  - 断言消息仍建议先赋变量、写成单行（`model/gpt.py:66` 的写法），保持既有风格；
  - 换机器若漏装 ruff / mypy，hook 会直接报找不到命令（`pip install -e ".[dev]"` 已含两者）。
- **detect-secrets 已从 hook 移除**（本机没装它的 CLI/模块，local 模式跑不了），
  `.pre-commit-config.yaml:31` 留了加回的注释；CONTRIBUTING.md 里写它还在，是旧描述。
- **产物不要入库**。注意 `.gitignore` **一行只能写一个模式**：写在一行上（如过去的
  `outputs/ runs/`）会被 git 当成一个带空格的模式、完全不生效。2026-10-02 修的就是这处——
  `outputs/` 原先其实没被忽略，OpenAI TF 权重的 475MB 差点混进提交。
  2026-10-02 已把全部同类问题一次拆行修好，并用 `git check-ignore -v` 逐条验证生效
  （目录型模式要拿真实目录测，对不存在的路径 `git check-ignore` 会报 NOT IGNORED，属误报）。
- 提交前无需手动格式化（ruff check --fix / ruff format / mypy 都由 hook 跑，用的是本机版本），
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
