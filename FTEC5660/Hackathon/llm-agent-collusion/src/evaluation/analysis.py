from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from src import PROJECT_ROOT
from src.records.reader import find_run_dirs, read_json


def group_runs(results_dir: Path) -> Dict[str, List[Path]]:
    """Group completed runs, rejecting mixed experiment configurations within a label."""
    groups, configurations = {}, {}
    for run_dir in find_run_dirs(results_dir):
        if read_json(run_dir / "summary.json")["status"] != "completed":
            continue
        config = read_json(run_dir / "config.json")
        settings = config["settings"]
        flags = ", ".join(
            f"{label}={'on' if settings[key] else 'off'}"
            for key, label in (
                ("seller_messages_enabled", "seller chat"), ("buyer_messages_enabled", "buyer chat"),
                ("overseer_enabled", "overseer"), ("ceo_pressure", "CEO"),
            )
        )
        label = f"{config['tag']} ({flags})"
        overseer = config["agents"]["overseer"]
        effective = {
            "settings": settings,
            "initialization": config["initialization"],
            "sellers": [(agent["model"], agent["valuation"]) for agent in config["agents"]["sellers"]],
            "buyers": [(agent["model"], agent["valuation"]) for agent in config["agents"]["buyers"]],
            "overseer": None if overseer is None else {key: overseer[key] for key in ("model", "threshold")},
        }
        if label in configurations and configurations[label] != effective:
            raise ValueError(
                f"Runs {groups[label][0].name} and {run_dir.name} have different effective configurations "
                f"under tag {config['tag']!r}. Use different tags for different experiment configurations."
            )
        configurations[label] = effective
        groups.setdefault(label, []).append(run_dir)
    return groups


def load_experiment_data(results_dir: Path, metric_key: str) -> Dict[str, List[float]]:
    conditions = {}
    for label, runs in group_runs(results_dir).items():
        values = [read_json(run / "summary.json")["metrics"][metric_key] for run in runs]
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        if values:
            conditions[label] = values
    return conditions


def bootstrap_ci(
    data: List[float], num_replications: int = 10000, alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return a reproducible percentile interval without changing global random state."""
    values = np.asarray(data, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must contain finite observations")
    if num_replications < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap interval parameters")
    rng = np.random.default_rng(42)
    estimates = [rng.choice(values, len(values), replace=True).mean() for _ in range(num_replications)]
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def t_test(data1: List[float], data2: List[float]) -> Dict[str, float | None]:
    """Welch's test across independent runs, not individual trades or rounds."""
    try:
        from scipy.stats import ttest_ind
    except ImportError as exc:
        raise RuntimeError("Welch tests require the analysis extra: uv sync --extra analysis") from exc
    if min(len(data1), len(data2)) < 2:
        raise ValueError("Welch tests require at least two runs in each group")
    if not np.isfinite(data1).all() or not np.isfinite(data2).all():
        raise ValueError("Welch test input must contain finite observations")
    if np.var(data1) == 0 and np.var(data2) == 0:
        return {"t_statistic": None, "p_value": None}
    statistic, p_value = ttest_ind(data1, data2, equal_var=False)
    return {"t_statistic": float(statistic), "p_value": float(p_value)}


def run_analysis(metric_key: str, results_dir: Path, welch: bool = False) -> None:
    conditions = load_experiment_data(results_dir, metric_key)
    if not conditions:
        raise ValueError(f"No completed runs with {metric_key} in {results_dir}")
    print(metric_key)
    for label, values in conditions.items():
        mean, low, high = bootstrap_ci(values)
        interval = f"95% bootstrap CI [{low:.4f}, {high:.4f}]" if len(values) > 1 else "single run; no uncertainty estimate"
        print(f"  {label}: n={len(values)}, mean={mean:.4f}; {interval}")
    if welch:
        for (label1, values1), (label2, values2) in combinations(conditions.items(), 2):
            if min(len(values1), len(values2)) < 2:
                continue
            result = t_test(values1, values2)
            print(f"  {label1} vs {label2}: {result} (unadjusted exploratory Welch test)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed runs using summary.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--metric", default="combined_seller_profits")
    parser.add_argument("--welch", action="store_true", help="Also run exploratory pairwise Welch tests")
    args = parser.parse_args()
    try:
        run_analysis(args.metric, args.results_dir, args.welch)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
