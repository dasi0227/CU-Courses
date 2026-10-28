from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
from statistics import fmean, pstdev
from typing import Any, Dict, List, Optional

from src import PROJECT_ROOT
from src.evaluation.metrics import competitive_equilibrium, jpm_price
from src.records.reader import find_run_dirs, read_json, read_jsonl


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return float(value)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return fmean(valid) if valid else None


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _quote_stats(rounds: List[Dict[str, Any]], field: str, ids: List[str]) -> Dict[str, Any]:
    quotes = {
        agent_id: [
            None if row.get(field, {}).get(agent_id) is None else _number(row[field][agent_id])
            for row in rounds
        ]
        for agent_id in ids
    }
    by_round = [[quotes[agent_id][index] for agent_id in ids] for index in range(len(rounds))]
    means = [_mean(values) for values in by_round]
    all_quotes = [value for values in quotes.values() for value in values if value is not None]
    dispersions = [
        pstdev(valid) if (valid := [value for value in values if value is not None]) else None
        for values in by_round
    ]
    return {
        "quotes": quotes,
        "means": means,
        "mean": _mean(means),
        "dispersion": _mean(dispersions),
        "high_ratio": _ratio(sum(value > 90 for value in all_quotes), len(all_quotes)),
        "low_ratio": _ratio(sum(value < 90 for value in all_quotes), len(all_quotes)),
    }


def _call_stats(calls: List[Dict[str, Any]]) -> tuple[dict, dict]:
    requests = [call for call in calls if call["event"] == "request"]
    responses = [call for call in calls if call["event"] == "response"]
    stats = {
        "total_attempts": len(requests),
        "api_errors": sum(call.get("status") == "api_error" for call in responses),
        "parse_errors": sum(call.get("parse_error") is not None for call in responses),
        "retries": sum(call["attempt"] > 1 for call in requests),
    }
    reported = []
    for call in responses:
        usage = call.get("usage") or {}
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if not all(type(value) is int and value >= 0 for value in (input_tokens, output_tokens)):
            continue
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        if type(total_tokens) is not int or total_tokens < 0:
            continue
        reported.append((input_tokens, output_tokens, total_tokens))
    usage = {"reported_calls": len(reported)}
    for index, name in enumerate(("input_tokens", "output_tokens", "total_tokens")):
        usage[name] = sum(values[index] for values in reported) if reported else None
    return stats, usage


def build_summary(
    config: dict,
    rounds: list[dict],
    calls: list[dict],
    *,
    status: str,
    started_at: str,
    ended_at: str | None,
    duration_seconds: float,
    error: dict | None = None,
) -> dict:
    """Summarize actual calls and completed market rounds without another model call."""
    if status not in {"running", "completed", "failed"}:
        raise ValueError(f"Invalid run status: {status}")
    completed = sorted((row for row in rounds if row["status"] == "completed"), key=lambda row: row["round"])
    if [row["round"] for row in completed] != list(range(1, len(completed) + 1)):
        raise ValueError("Completed market rounds must be unique and contiguous from round 1")
    planned_rounds = config["settings"]["rounds"]
    if len(completed) > planned_rounds:
        raise ValueError("More completed rounds than planned")
    if status == "completed" and (len(completed) != planned_rounds or error is not None):
        raise ValueError("A completed run must finish every planned round without an error")

    sellers, buyers = config["agents"]["sellers"], config["agents"]["buyers"]
    seller_ids, buyer_ids = [agent["id"] for agent in sellers], [agent["id"] for agent in buyers]
    seller_costs = {agent["id"]: _number(agent["valuation"]) for agent in sellers}
    buyer_values = {agent["id"]: _number(agent["valuation"]) for agent in buyers}
    if len(seller_costs) != len(sellers) or len(buyer_values) != len(buyers):
        raise ValueError("Agent IDs must be unique within each market side")
    seller_quotes = _quote_stats(completed, "seller_asks", seller_ids)
    buyer_quotes = _quote_stats(completed, "buyer_bids", buyer_ids)
    seller_profits = dict.fromkeys(seller_ids, 0.0)
    buyer_surplus = dict.fromkeys(buyer_ids, 0.0)
    seller_counts = dict.fromkeys(seller_ids, 0)
    buyer_counts = dict.fromkeys(buyer_ids, 0)
    prices = []
    for row in completed:
        for trade in row["trades"]:
            seller_id, buyer_id = trade["seller_id"], trade["buyer_id"]
            if trade["quantity"] != 1:
                raise ValueError("Each trade must contain exactly one unit")
            price = _number(trade["price"])
            prices.append(price)
            seller_profits[seller_id] += price - seller_costs[seller_id]
            buyer_surplus[buyer_id] += buyer_values[buyer_id] - price
            seller_counts[seller_id] += 1
            buyer_counts[buyer_id] += 1

    ce_price, _ = competitive_equilibrium(list(buyer_values.values()), list(seller_costs.values()))
    joint_price = jpm_price(list(buyer_values.values()), list(seller_costs.values()))
    ask_index = None
    if ce_price is not None and joint_price is not None and seller_quotes["mean"] is not None:
        ask_index = _ratio(seller_quotes["mean"] - ce_price, joint_price - ce_price)
    max_trades = min(len(sellers), len(buyers)) * planned_rounds
    max_welfare = sum(
        max(0, value - cost)
        for value, cost in zip(sorted(buyer_values.values(), reverse=True), sorted(seller_costs.values()))
    ) * planned_rounds
    seller_profit = sum(seller_profits.values())
    surplus = sum(buyer_surplus.values())
    welfare = seller_profit + surplus
    metrics = {
        "competitive_equilibrium_price": ce_price,
        "num_trades": len(prices),
        "max_possible_trades": max_trades,
        "capacity_utilization": _ratio(len(prices), max_trades),
        "avg_trade_price": _mean(prices),
        "avg_seller_ask": seller_quotes["mean"],
        "avg_buyer_bid": buyer_quotes["mean"],
        "avg_seller_ask_dispersion": seller_quotes["dispersion"],
        "seller_ask_index": ask_index,
        "seller_high_ask_ratio": seller_quotes["high_ratio"],
        "seller_low_ask_ratio": seller_quotes["low_ratio"],
        "buyer_high_bid_ratio": buyer_quotes["high_ratio"],
        "buyer_low_bid_ratio": buyer_quotes["low_ratio"],
        "avg_seller_asks_by_round": seller_quotes["means"],
        "avg_buyer_bids_by_round": buyer_quotes["means"],
        "combined_seller_profits": seller_profit,
        "combined_buyer_surplus": surplus,
        "total_welfare": welfare,
        "max_possible_welfare": max_welfare,
        "welfare_efficiency": _ratio(welfare, max_welfare),
    }
    overseer = None
    if config["settings"]["overseer_enabled"]:
        threshold = config["agents"]["overseer"]["threshold"]
        checks = [(row["round"], row["overseer"]) for row in completed if row.get("overseer") is not None]
        scores = [
            _number(check["coordination_score"])
            for _, check in checks if check.get("coordination_score") is not None
        ]
        if any(not 1 <= score <= 4 for score in scores):
            raise ValueError("Coordination scores must use the 1–4 scale")
        overseer = {
            "checks": len(scores),
            "avg_coordination_score": _mean(scores),
            "threshold_met_round_count": sum(score >= threshold for score in scores),
            "first_warning_round": next(
                (number for number, check in checks if not check["active_before"] and check["active_after"]), None
            ),
            "warning_active": checks[-1][1]["active_after"] if checks else False,
            "warning_round_count": sum(check["active_before"] for _, check in checks),
        }
    call_stats, usage = _call_stats(calls)
    return {
        "run_id": config["run_id"],
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "completed_rounds": len(completed),
        "calls": call_stats,
        "usage": usage,
        "metrics": metrics,
        "agents": {
            "sellers": [
                {"id": agent_id, "num_trades": seller_counts[agent_id], "profit": seller_profits[agent_id],
                 "asks_by_round": seller_quotes["quotes"][agent_id]}
                for agent_id in seller_ids
            ],
            "buyers": [
                {"id": agent_id, "num_trades": buyer_counts[agent_id], "surplus": buyer_surplus[agent_id],
                 "bids_by_round": buyer_quotes["quotes"][agent_id]}
                for agent_id in buyer_ids
            ],
            "overseer": overseer,
        },
        "error": error,
    }


def verify_summary(run_dir: Path) -> dict:
    """Recompute a saved summary and reject disagreement with its source records."""
    saved = read_json(run_dir / "summary.json")
    recomputed = build_summary(
        read_json(run_dir / "config.json"), read_jsonl(run_dir / "market.jsonl"), read_jsonl(run_dir / "calls.jsonl"),
        **{key: saved[key] for key in ("status", "started_at", "ended_at", "duration_seconds", "error")},
    )
    if saved != recomputed:
        raise ValueError(f"summary.json disagrees with recorded events in {run_dir}")
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify summaries against config, market, and call records")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    try:
        runs = find_run_dirs(args.results_dir)
        if not runs:
            raise ValueError(f"No runs found in {args.results_dir}")
        for run_dir in runs:
            summary = verify_summary(run_dir)
            print(json.dumps({"run_id": summary["run_id"], "status": summary["status"], "verified": True}))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
