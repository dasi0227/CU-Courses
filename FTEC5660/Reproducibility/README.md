# README

## Introduction

This project presents a small-scale reproduction of Agentic Context Engineering (ACE) on the Formula financial reasoning benchmark. I compare three settings: an empty-playbook baseline, ACE with a learned playbook, and a JEV-based variant that selects the top-10 relevant playbook entries for each question. Using DeepSeek-V3.1-Terminus, I repeat each experiment three times and measure answer accuracy, token usage, and runtime to examine the effects of learned memory and selective retrieval.

- Repository：[ace-agent/ace](https://github.com/ace-agent/ace)
- Thesis：[Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)

## Environment Preparation

```shell
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. install dependencies
uv sync --locked --python 3.12
# 3. check environment
.venv/bin/python --version
uv sync --locked --check --python 3.12
uv pip check --python .venv/bin/python
```

> Please first navigate to the `ace/` subdirectory.

## API Preparation

1. Create `.env` file to the root directory: `cp -n .env.example .env`

2. Get your DeepSeek API key from: [BaiLian Console](https://bailian.console.aliyun.com/cn-beijing/?tab=app#/api-key)

3. Get your TypeSafe API key from: [TypeSafe Console](https://console.typesafe.ai/)

4. Replace the placeholder value with your own API Key.


## Quick Start

**查看参数、生成计划：**

```bash
.venv/bin/python -m eval.experiments --help
```

默认运行等同 `--dry-run`，只生成或校验 `plan.json`，不调用模型 API。

**正式运行示例，每组仅执行第 1 次：**

```bash
# A：直接测试
.venv/bin/python -u -m eval.experiments \
  --execute --config config.json \
  --groups A --trial 1 --output results-rerun

# B：独立学习手册，再测试
.venv/bin/python -u -m eval.experiments \
  --execute --config config.json \
  --groups B --trial 1 --output results-rerun

# C：B/01 正常完成后，复用其最佳手册
.venv/bin/python -u -m eval.experiments \
  --execute --config config.json \
  --groups C --trial 1 --output results-rerun
```

- 只有 `--execute` 会启用真实模型 API；上述正式命令会消耗 API 用量。
- 将 `--trial 1` 改为 `2` 或 `3`，运行对应编号；省略 `--trial` 会运行该组全部三次。
- B/C 必须使用同一输出根目录；`C/02` 只复用正常完成的 `B/02` 最佳手册，其余同理。
- 已有 `results/` 保存归档结果，示例统一使用独立的 `results-rerun/`。另一轮实验应更换为新的输出根目录；已有组和编号目录会被拒绝覆盖，当前入口不自动断点续跑。

**离线检查已有计划，不调用 API：**

```bash
.venv/bin/python -m eval.experiments \
  --dry-run --config config.json --output results
```

该命令检查当前配置与样本 ID 是否与已有计划一致；已有计划不会被改写。C 正式执行前还会检查对应 B 的组名、重复编号和完成状态，并读取其最佳手册。

## 目录说明

```text
ace/
├── .env.example                      # API 变量模板
├── .env                              # 本地密钥，不提交
├── pyproject.toml                    # 项目与依赖声明
├── uv.lock                           # 锁定依赖版本
├── src/                              # ACE 核心、角色提示词及公共工具
│   └── experiments/                  # A/B/C 编排、JEV、日志及运行检查
├── eval/                             # Formula 入口、数据处理及数据集
├── config.json                       # 实验配置
├── docs/                             # 框架图片及详细实验说明
└── results/                          # 已归档结果；新输出根使用相同结构
    ├── plan.json                     # 配置快照及训练、验证、测试样本 ID
    ├── summary.csv                   # 各组统计、耗时和用量汇总
    ├── A/
    │   └── {01,02,03}/               # 三个独立重复目录，下同
    │       ├── predictions.jsonl     # 每题预测、标准答案、正误和状态
    │       ├── llm_calls.jsonl       # DeepSeek 提示词、回答及调用用量
    │       └── result.json           # 本次指标、耗时、累计用量及来源
    ├── B/
    │   └── {01,02,03}/
    │       ├── predictions.jsonl
    │       ├── llm_calls.jsonl       # 包含训练、验证和测试阶段
    │       ├── result.json
    │       └── training/
    │           ├── best_playbook.txt # 验证集选出的最佳手册，供 C 复用
    │           └── train_results.json # 学习步骤、验证记录及最佳检查点
    └── C/
        └── {01,02,03}/
            ├── predictions.jsonl
            ├── llm_calls.jsonl
            ├── result.json          # 另含 B 手册来源及共享学习用量
            └── jev_audit.jsonl      # JEV 请求、评分、top-10 选择及用量
```

`results/` 根目录只保留 `plan.json`、`summary.csv` 和 A/B/C。`plan.json` 仅保存 `config` 配置快照及 `ids` 中的训练、验证、测试样本 ID；数据路径、抽样设置等保留在配置快照中，不重复记录。运行入口核对配置和 ID 是否与已有计划一致；运行组由 `--groups` 决定，所需密钥按所选组在运行时检查。`training.max_num_rounds` 设置 B 的反思上限，当前为 3；A/C 不训练。是否调用 API 由显式 `--execute` 决定。计划不保留历史快照，不生成版本化计划、版本化汇总或 `invocations` 文件。A/02、A/03 的 `llm_calls.jsonl` 为每题最终回答视图，`result.json` 中分别保存最终回答用量与实际累计用量。

更详细的参数及文件字段见 [ace/docs/formula_experiments.txt](ace/docs/formula_experiments.txt)。
