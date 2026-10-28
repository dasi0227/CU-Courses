from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from statistics import fmean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src import PROJECT_ROOT
from src.evaluation.analysis import group_runs
from src.records.reader import read_jsonl


PLOT_METRICS = {
    "avg_seller_ask": "Seller ask price",
    "avg_buyer_bid": "Buyer bid price",
    "avg_trade_price": "Trade price",
    "ask_dispersion": "Within-round seller ask dispersion",
}


def load_round_metrics(run_dir: Path) -> Dict[int, Dict[str, float | None]]:
    """Derive plotting series from actual completed market records."""
    result = {}
    for row in read_jsonl(run_dir / "market.jsonl"):
        if row["status"] != "completed":
            continue
        asks = [value for value in row["seller_asks"].values() if value is not None]
        bids = [value for value in row["buyer_bids"].values() if value is not None]
        prices = [trade["price"] for trade in row["trades"]]
        result[row["round"]] = {
            "avg_seller_ask": fmean(asks) if asks else None,
            "avg_buyer_bid": fmean(bids) if bids else None,
            "avg_trade_price": fmean(prices) if prices else None,
            "ask_dispersion": pstdev(asks) if asks else None,
        }
    return result


def aggregate_round_metrics(runs: List[Path], metric: str, num_rounds: int | None = None) -> tuple:
    series = [load_round_metrics(run) for run in runs]
    rounds = sorted({number for values in series for number in values if num_rounds is None or number <= num_rounds})
    means, margins = [], []
    for number in rounds:
        values = [data[number][metric] for data in series if number in data and data[number][metric] is not None]
        means.append(fmean(values) if values else np.nan)
        margins.append(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan)
    return np.array(rounds), np.array(means), np.array(margins)


def create_market_figure(results_dir: Path, output_dir: Path, num_rounds: int | None = None) -> Path:
    groups = group_runs(results_dir)
    if not groups:
        raise ValueError(f"No completed runs found in {results_dir}")
    if num_rounds is not None and num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"}):
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for ax, (metric, title) in zip(axes.flat, PLOT_METRICS.items()):
            for label, runs in groups.items():
                numbers, means, margins = aggregate_round_metrics(runs, metric, num_rounds)
                line, = ax.plot(numbers, means, marker="o", markersize=3, label=f"{label}; n={len(runs)}")
                ax.fill_between(numbers, means - margins, means + margins, color=line.get_color(), alpha=0.15)
            ax.set(title=title, xlabel="Round")
            ax.grid(alpha=0.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", fontsize=8)
        figure.suptitle("Completed runs: mean across runs; shading is ±1.96 SE where n > 1")
        figure.tight_layout(rect=(0, 0.06 + 0.02 * len(groups), 1, 0.95))
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "market_trends.png"
        try:
            figure.savefig(destination, dpi=200, bbox_inches="tight")
        finally:
            plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completed experiments from market.jsonl")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--num-rounds", type=int)
    args = parser.parse_args()
    try:
        print(create_market_figure(args.results_dir, args.output_dir, args.num_rounds))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
