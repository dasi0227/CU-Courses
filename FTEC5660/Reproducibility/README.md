# README

## Project Introduction

This project presents **a small-scale reproduction of Agentic Context Engineering on the Formula financial reasoning benchmark**. The project compares three settings: an empty-playbook baseline, ACE with a learned playbook, and a JEV-based variant that selects the top-10 relevant playbook entries for each question. Using DeepSeek-V3.1-Terminus, the project repeats each experiment three times and measure answer accuracy, token usage, and runtime to examine the effects of learned memory and selective retrieval.

- Repository：[ace-agent/ace](https://github.com/ace-agent/ace)
- Thesis：[Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)

## Environment Preparation

1. Install uv.

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install dependencies.

    ```shell
    uv sync --locked --python 3.12
    ```

3. Check environment.

    ```shell
    .venv/bin/python --version
    uv sync --locked --check --python 3.12
    uv pip check --python .venv/bin/python
    ```

4. Create `.env` file to the root directory.

    ```shell
    cp -n .env.example .env
    ```

5. Replace the placeholder value with your own API Key from [BaiLian Console](https://bailian.console.aliyun.com/cn-beijing/?tab=app#/api-key) and [TypeSafe Console](https://console.typesafe.ai/).

    ```shell
    DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
    TYPESAFE_API_KEY=YOUR_TYPESAFE_KEY
    ```

> Please first navigate to the `ace/` subdirectory.


## Quick Start

1. View the experiment script’s usage instructions.

    ```shell
    .venv/bin/python -m eval.experiments --help
    ```

2. Generate the experiment plan.

    ```shell
    .venv/bin/python -m eval.experiments --output results-rerun
    ```

3. Run the first trial of Group A.

    ```shell
    .venv/bin/python -u -m eval.experiments \
      --execute \
      --config config.json \
      --output results-rerun \
      --groups A --trial 1
    ```

4. Run the first trial of Group B.

    ```shell
    .venv/bin/python -u -m eval.experiments \
      --execute \
      --config config.json \
      --output results-rerun \
      --groups B --trial 1
    ```

5. Run the first trial of Group C.

    ```shell
    .venv/bin/python -u -m eval.experiments \
      --execute \
      --config config.json \
      --output results-rerun \
      --groups C --trial 1
    ```

## Directory Structure

```text
ace/
├── pyproject.toml                      # Project metadata and dependencies
├── uv.lock                             # Locked dependency versions
├── config.json                         # Experiment settings
├── src/
│   ├── ace.py                          # Sequential ACE learning and evaluation
│   ├── ace_batch.py                    # Batched ACE learning and evaluation
│   ├── core/                           # ACE generation, reflection, and curation
│   │   ├── generator.py                # Generate answers using the playbook
│   │   ├── reflector.py                # Diagnose mistakes and tag useful entries
│   │   └── curator.py                  # Propose and apply playbook updates
│   ├── prompts/                        # Prompt templates for the three roles
│   │   ├── generator.py                # Answer-generation prompt
│   │   ├── reflector.py                # Reflection prompts with/without labels
│   │   └── curator.py                  # Playbook-update and aggregation prompts
│   ├── common/                         # Shared model and evaluation utilities
│   │   ├── llm.py                      # Model calls, timing, and error handling
│   │   ├── logger.py                   # Call, entry-usage, and update logging
│   │   ├── playbook_utils.py           # Parse, update, and extract playbook entries
│   │   └── utils.py                    # Clients, answer parsing, and evaluation
│   └── experiments/                    # A/B/C orchestration and result handling
│       ├── data.py                     # Validate config and select Formula samples
│       ├── runner.py                   # CLI, trials, B-to-C reuse, and summaries
│       ├── runtime.py                  # Audited model calls, retries, and usage
│       └── jev.py                      # JEV scoring and stable top-10 selection
├── eval/
│   ├── experiments.py                  # Entry point for A/B/C experiments
│   ├── data_processor.py               # Parse tasks and score answers
│   └── data/                           # Formula training, validation, and test datasets
└── results/                            # Saved results
    ├── plan.json                       # Config snapshot and train/val/test sample IDs
    ├── summary.csv                     # Accuracy statistics, runtime, and token usage
    ├── A/{01,02,03}/                   # Three independent trials per group
    │   ├── predictions.jsonl           # Per-question predictions, labels, and status
    │   ├── llm_calls.jsonl             # DeepSeek prompts, responses, and usage
    │   └── result.json                 # Trial metrics, runtime, and cumulative usage
    ├── B/{01,02,03}/
    │   ├── predictions.jsonl
    │   ├── llm_calls.jsonl
    │   ├── result.json
    │   ├── best_playbook.txt           # Best validation playbook
    │   └── train_results.json          # Learning steps, validation, and best checkpoint
    └── C/{01,02,03}/
        ├── predictions.jsonl
        ├── llm_calls.jsonl
        ├── result.json
        └── jev_audit.jsonl             # JEV requests, scores, top-10 choices, and usage
```
