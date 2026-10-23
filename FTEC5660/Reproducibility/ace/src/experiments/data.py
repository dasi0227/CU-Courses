"""Read-only Formula data preparation and experiment configuration validation.

This module uses only the standard library. It never preprocesses Formula inputs,
loads API keys, writes datasets, or constructs a model client.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
from typing import Any
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
GROUPS = ("A", "B", "C")
FORMULA_PATTERN = re.compile(r"^Use formula (.*?) to answer the question\.")
BAILIAN_MODEL = "siliconflow/deepseek-v3.1-terminus"
BAILIAN_KEY_ENV = "DEEPSEEK_API_KEY"


def _integer(value: Any, name: str, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be >= {minimum}{upper}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return float(value)


def validate_llm_config(llm: dict[str, Any]) -> dict[str, Any]:
    """Pin the same Beijing/non-thinking backend for all three experiment arms.

    DEEPSEEK_API_KEY is the user's local variable name for the Bailian key.
    It does not select DeepSeek's own API. Check the destination before reading
    that credential, so a stale SambaNova configuration cannot receive it.
    """
    if not isinstance(llm, dict):
        raise ValueError("llm must be an object")
    if llm.get("api_provider") != "dashscope":
        raise ValueError("llm.api_provider must be dashscope for the Bailian protocol")
    if llm.get("model") != BAILIAN_MODEL:
        raise ValueError(f"llm.model must be {BAILIAN_MODEL}")
    if llm.get("api_key_env") != BAILIAN_KEY_ENV:
        raise ValueError(f"llm.api_key_env must be {BAILIAN_KEY_ENV}")
    if llm.get("enable_thinking") is not False:
        raise ValueError("llm.enable_thinking must be false")
    if llm.get("region") != "cn-beijing":
        raise ValueError("llm.region must be cn-beijing for this SiliconFlow service")
    base_url = llm.get("base_url")
    if not isinstance(base_url, str):
        raise ValueError("llm.base_url must be a Beijing Bailian OpenAI-compatible URL")
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    valid_host = host == "dashscope.aliyuncs.com" or (
        host.endswith(".cn-beijing.maas.aliyuncs.com")
        and host != ".cn-beijing.maas.aliyuncs.com"
    )
    if (parsed.scheme != "https" or not valid_host or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/compatible-mode/v1"
            or "{" in host or "}" in host):
        raise ValueError("llm.base_url must be the shared or workspace-specific Beijing Bailian endpoint")
    _integer(llm.get("max_tokens"), "llm.max_tokens")
    if isinstance(llm.get("temperature"), bool) or llm.get("temperature") != 0:
        raise ValueError("llm.temperature must be 0 for every group")
    _number(llm.get("timeout"), "llm.timeout", 0.001)
    _integer(llm.get("retries"), "llm.retries", 0)
    _number(llm.get("sleep_seconds"), "llm.sleep_seconds", 0)
    return llm


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the settings used by all experiment runners, without side effects."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    for name in ("seed", "repeats", "llm", "training", "data", "jev"):
        if name not in config:
            raise ValueError(f"missing configuration field: {name}")
    _integer(config["seed"], "seed", 0)
    _integer(config["repeats"], "repeats", 3, 5)
    for name in ("llm", "training", "data", "jev"):
        if not isinstance(config[name], dict):
            raise ValueError(f"{name} must be an object")

    validate_llm_config(config["llm"])

    training = config["training"]
    for name in ("num_epochs", "eval_steps", "save_steps", "curator_frequency", "batch_size"):
        _integer(training.get(name), f"training.{name}")
    if training["batch_size"] != 1:
        raise ValueError("training.batch_size must be 1 to preserve the ACE sample-wise workflow")
    rounds = _integer(training.get("max_num_rounds"), "training.max_num_rounds")
    if rounds != 3:
        raise ValueError("training.max_num_rounds must be 3")

    data = config["data"]
    for split in SPLITS:
        name = f"{split}_path"
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ValueError(f"data.{name} must be a non-empty path")
    sampling = data.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("data.sampling must be an object")
    mode = sampling.get("mode")
    if mode not in ("per_formula", "all"):
        raise ValueError("data.sampling.mode must be per_formula or all")
    if mode == "per_formula":
        for split in ("train", "val"):
            _integer(sampling.get(f"{split}_per_formula"), f"data.sampling.{split}_per_formula")
        if (sampling["train_per_formula"], sampling["val_per_formula"]) != (2, 1):
            raise ValueError("per_formula sampling requires 2 training and 1 validation sample per formula")

    jev = config["jev"]
    for name in ("base_url", "model", "api_key_env"):
        if not isinstance(jev.get(name), str) or not jev[name].strip():
            raise ValueError(f"jev.{name} must be a non-empty string")
    if _integer(jev.get("k"), "jev.k") != 10:
        raise ValueError("jev.k must be fixed at 10")
    if "k_candidates" in jev:
        raise ValueError("jev.k_candidates is not supported; C uses fixed top-10 without validation tuning")
    _integer(jev.get("batch_size"), "jev.batch_size")
    _number(jev.get("timeout"), "jev.timeout", 0.001)

    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON settings; dataset paths are resolved by prepare_data at project root."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return validate_config(config)


read_config = load_config


def _read_split(path: Path, split: str) -> list[dict[str, Any]]:
    content = path.read_bytes()
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(source, dict) or not isinstance(source.get("context"), str) or "target" not in source:
            raise ValueError(f"{path}:{line_number}: expected context string and target")
        match = FORMULA_PATTERN.match(source["context"])
        if match is None or not match.group(1).strip():
            raise ValueError(f"{path}:{line_number}: could not identify formula name")
        rows.append({
            "id": f"{split}:{line_number:06d}",
            "source_split": split,
            "source_line": line_number,
            "formula_name": match.group(1),
            "raw_context": source["context"],
            "context": source["context"],
            "target": source["target"],
        })
    if not rows:
        raise ValueError(f"{path}: dataset must not be empty")
    return rows


def _sample_per_formula(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    by_formula: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_formula.setdefault(row["formula_name"], []).append(row)
    rng = random.Random(seed)
    selected = []
    for name in sorted(by_formula):
        if len(by_formula[name]) < count:
            raise ValueError(f"formula {name!r} has fewer than {count} samples")
        selected.extend(rng.sample(by_formula[name], count))
    return sorted(selected, key=lambda sample: sample["source_line"])


def prepare_data(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Return the selected train, validation and test samples without writing files."""
    validate_config(config)
    project_root = Path(project_root).expanduser().resolve()
    result: dict[str, Any] = {}
    sampling = config["data"]["sampling"]
    for split_index, split in enumerate(SPLITS):
        path = Path(config["data"][f"{split}_path"]).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        rows = _read_split(path, split)
        split_seed = config["seed"] + split_index
        selected = rows
        if sampling["mode"] == "per_formula" and split != "test":
            selected = _sample_per_formula(rows, sampling[f"{split}_per_formula"], split_seed)
        result[split] = selected
    formula_sets = [{row["formula_name"] for row in result[split]} for split in SPLITS]
    if len(formula_sets[0]) != 50 or not all(names == formula_sets[0] for names in formula_sets):
        raise ValueError("Formula datasets must contain the same 50 formulas in every split")
    expected_counts = ((50 * sampling['train_per_formula'], 50 * sampling['val_per_formula'], 200)
                       if sampling['mode'] == 'per_formula' else (500, 300, 200))
    if tuple(len(result[split]) for split in SPLITS) != expected_counts:
        raise ValueError(f"{sampling['mode']} sampling requires split sizes {expected_counts}")
    return result
