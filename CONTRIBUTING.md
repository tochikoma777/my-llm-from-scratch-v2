# Contributing

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
make install     # pip install -e ".[dev,viz]" + pre-commit install
make check       # lint + test，提 PR 前必须绿
```

## 日常命令

| 目标 | 作用 |
|---|---|
| `make lint` | `ruff check src tests scripts` + `mypy src` |
| `make fmt` | `ruff format src tests scripts` |
| `make test` | `pytest -q`（默认跳过 `slow`） |
| `make test-full` | 跑 parity 测试，会下载 GPT-2 权重 |
| `make demo` | tiny config 训练 + 生成，冒烟用 |
| `make clean` | 清缓存 |

## 硬规则

1. **`GPTConfig` 是唯一配置来源**。禁止在函数体内写超参字面量——v1 的痛点就是把
   `lr/epochs/batch_size` 埋在 `__main__` 里（`TRAIN:530-546`），既不能 import 也不能测。
   新超参先进 `GPTConfig` 或对应 yaml。
2. **类型注解必须能通过 `mypy --strict`**。`tests/` 可以不注解，但 `src/` 不行。
3. **不要引入 CWD 相对路径**。所有落盘路径走参数注入 + `pathlib`，输出统一到 `outputs/`（已 gitignore）。
4. **改动 `model/` 必须同步 parity 测试**。任何影响数值的改动（`model/` 下所有文件）都要有对应断言，
   否则按 breaking change 处理。
5. **parity 测试必须标记 `@pytest.mark.slow`**，保证日常 `pytest` 不联网。
6. **`tensorflow` 不得进入主依赖**。它只用于 `weights/openai_tf.py` 兼容层，属可选依赖。

## 搬运 v1 代码时

只允许从 `../v1-reference` **只读复制**，不要改它。搬运到 `src/my_llm/` 后：

- 保留算法逻辑，重组织方式（扁平字典 config → `GPTConfig`、扁平导入 → 包导入）。
- 必须顺手修掉 `docs/00-现状盘点.md` 第四部分列出的缺陷，并在 PR 里说明改了哪一项。
- 从零手写 / 改动的部分要有注释说明来源文件与行号，便于日后回溯。

## 提交与 PR

- `pre-commit` 会跑 ruff check --fix / ruff format / mypy（`repo: local` + `language: system`，
  用的是本机已装的 ruff / mypy）；不要 `git commit --no-verify`。
  hook 改写文件后重跑一次提交即可，第一次提交失败是正常的。
- **detect-secrets 当前未启用**（本机未安装 detect-secrets，改 local 模式后 hook 必然失败）。
  需要密钥扫描时先 `pip install detect-secrets`，再把 `.pre-commit-config.yaml` 注释里
  那段 hook 配置加回来。当前配置见 `.pre-commit-config.yaml`。
- 不要把权重、数据集、loss 曲线 PDF 提交到仓库（`.gitignore` 已覆盖）。
- PR 描述里写清：改了什么、是否影响数值、parity 是否通过。
