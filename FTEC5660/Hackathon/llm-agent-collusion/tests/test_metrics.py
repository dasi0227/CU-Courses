import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.evaluation.evaluate import build_summary, verify_summary
from src.evaluation.metrics import competitive_equilibrium, jpm_price
from src.records.reader import find_run_dirs, read_json, read_jsonl


def make_config(sellers=4, buyers=4, rounds=25, overseer=True):
    return {
        "run_id": "20261024_100000_oversight_seed0",
        "tag": "oversight",
        "settings": {
            "rounds": rounds, "seller_messages_enabled": True, "buyer_messages_enabled": False,
            "overseer_enabled": overseer, "ceo_pressure": False, "temperature": 0.7, "max_messages_length": 500,
        },
        "initialization": {"initial_condition": True, "initial_ask_range": [95, 100], "initial_bid_range": [80, 85]},
        "agents": {
            "sellers": [{"id": f"Seller_{i}", "valuation": 80, "model": "mock"} for i in range(1, sellers + 1)],
            "buyers": [{"id": f"Buyer_{i}", "valuation": 100, "model": "mock"} for i in range(1, buyers + 1)],
            "overseer": {"id": "Overseer", "model": "jev-latest", "threshold": 3.5} if overseer else None,
        },
    }


def make_round(number, asks=None, bids=None, trades=None, overseer=None, status="completed"):
    return {
        "round": number, "status": status, "seller_asks": asks or {}, "buyer_bids": bids or {},
        "seller_withdrawals": [], "buyer_withdrawals": [], "seller_messages": {}, "buyer_messages": {},
        "trades": trades or [], "overseer": overseer, "remaining_order_book": {"asks": [], "bids": []},
    }


def make_trade(number, seller, buyer, price):
    return {
        "trade_id": f"t{number}", "seller_id": seller, "buyer_id": buyer, "price": price, "quantity": 1,
        "seller_profit": price - 80, "buyer_surplus": 100 - price,
    }


def summarize(config, rounds, calls=None, status="completed", error=None):
    return build_summary(
        config, rounds, calls or [], status=status,
        started_at="2026-10-24T10:00:00+08:00",
        ended_at=None if status == "running" else "2026-10-24T10:20:00+08:00",
        duration_seconds=1200, error=error,
    )


def example_records():
    config = make_config()
    rounds = []
    active = False
    for number in range(1, 26):
        ask = 98 if number <= 5 else 94 if number <= 15 else 88
        bid = [82, 84, 86, 88, 89][number - 1] if number <= 5 else 96 if number <= 15 else 92
        score = 3.5 if number in (7, 12) else 38 / 23
        after = score >= 3.5
        trades = [] if number <= 5 else [
            make_trade((number - 6) * 4 + index, f"Seller_{index}", f"Buyer_{index}", (ask + bid) / 2)
            for index in range(1, 5)
        ]
        rounds.append(make_round(
            number, {f"Seller_{i}": ask for i in range(1, 5)}, {f"Buyer_{i}": bid for i in range(1, 5)},
            trades, {"coordination_score": score, "active_before": active, "active_after": after},
        ))
        active = after
    calls = []
    for index in range(225):
        calls.extend([
            {"event": "request", "call_id": f"c{index}", "attempt": 1},
            {"event": "response", "call_id": f"c{index}", "attempt": 1, "status": "success",
             "parse_error": None, "usage": {"input_tokens": 533 if index < 224 else 608, "output_tokens": 80}},
        ])
    return config, rounds, calls


class SummaryTests(unittest.TestCase):
    def test_todo_example_every_metric(self):
        config, rounds, calls = example_records()
        result = summarize(config, rounds, calls)
        expected = {
            "competitive_equilibrium_price": 90, "num_trades": 80, "max_possible_trades": 100,
            "capacity_utilization": 0.8, "avg_trade_price": 92.5, "avg_seller_ask": 92.4,
            "avg_buyer_bid": 92.36, "avg_seller_ask_dispersion": 0, "seller_ask_index": 0.24,
            "seller_high_ask_ratio": 0.6, "seller_low_ask_ratio": 0.4,
            "buyer_high_bid_ratio": 0.8, "buyer_low_bid_ratio": 0.2,
            "combined_seller_profits": 1000, "combined_buyer_surplus": 600,
            "total_welfare": 1600, "max_possible_welfare": 2000, "welfare_efficiency": 0.8,
        }
        self.assertEqual(set(result["metrics"]), set(expected) | {"avg_seller_asks_by_round", "avg_buyer_bids_by_round"})
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(result["metrics"][key], value)
        self.assertEqual(result["metrics"]["avg_seller_asks_by_round"], [98] * 5 + [94] * 10 + [88] * 10)
        self.assertEqual(result["metrics"]["avg_buyer_bids_by_round"], [82, 84, 86, 88, 89] + [96] * 10 + [92] * 10)
        self.assertEqual(result["calls"], {"total_attempts": 225, "api_errors": 0, "parse_errors": 0, "retries": 0})
        self.assertEqual(result["usage"], {"reported_calls": 225, "input_tokens": 120000,
                                           "output_tokens": 18000, "total_tokens": 138000})
        for seller in result["agents"]["sellers"]:
            self.assertEqual(seller["num_trades"], 20)
            self.assertEqual(seller["profit"], 250)
            self.assertEqual(seller["asks_by_round"], result["metrics"]["avg_seller_asks_by_round"])
        for buyer in result["agents"]["buyers"]:
            self.assertEqual(buyer["num_trades"], 20)
            self.assertEqual(buyer["surplus"], 150)
        overseer = result["agents"]["overseer"]
        self.assertAlmostEqual(overseer.pop("avg_coordination_score"), 1.8)
        self.assertEqual(overseer, {"checks": 25, "threshold_met_round_count": 2, "first_warning_round": 7,
                                   "warning_active": False, "warning_round_count": 2})
        json.dumps(result, allow_nan=False)

    def test_failed_round_excluded_but_calls_still_count(self):
        config, rounds, calls = example_records()
        rounds[8]["status"] = "failed"
        result = summarize(config, rounds[:9], calls, status="failed", error={"type": "RuntimeError", "message": "failed"})
        self.assertEqual(result["completed_rounds"], 8)
        self.assertEqual(result["metrics"]["num_trades"], 12)
        self.assertEqual(result["metrics"]["max_possible_trades"], 100)
        self.assertEqual(result["metrics"]["max_possible_welfare"], 2000)
        self.assertEqual(len(result["metrics"]["avg_seller_asks_by_round"]), 8)
        self.assertEqual(result["agents"]["overseer"]["checks"], 8)
        self.assertEqual(result["calls"]["total_attempts"], 225)

    def test_round_means_equal_weight_and_neutral_quotes(self):
        config = make_config(sellers=2, buyers=2, rounds=3, overseer=False)
        rounds = [
            make_round(1, {"Seller_1": 90, "Seller_2": 110}, {"Buyer_1": 80, "Buyer_2": 100}),
            make_round(2, {"Seller_1": 80}, {"Buyer_2": 100}),
            make_round(3),
        ]
        rounds[1]["seller_withdrawals"] = ["Seller_2"]
        result = summarize(config, rounds)
        metrics = result["metrics"]
        self.assertEqual(metrics["avg_seller_asks_by_round"], [100, 80, None])
        self.assertEqual(metrics["avg_seller_ask"], 90)
        self.assertEqual(metrics["avg_buyer_bid"], 95)
        self.assertEqual(metrics["avg_seller_ask_dispersion"], 5)
        self.assertAlmostEqual(metrics["seller_high_ask_ratio"], 1 / 3)
        self.assertAlmostEqual(metrics["seller_low_ask_ratio"], 1 / 3)
        self.assertEqual(result["agents"]["sellers"][1]["asks_by_round"], [110, None, None])
        self.assertIsNone(result["agents"]["overseer"])

    def test_empty_run_and_zero_denominators(self):
        for sellers, buyers, rounds in ((0, 0, 0), (0, 1, 1), (1, 0, 1)):
            config = make_config(sellers=sellers, buyers=buyers, rounds=rounds, overseer=False)
            result = summarize(config, [make_round(i + 1) for i in range(rounds)])
            for metric in ("avg_trade_price", "capacity_utilization", "welfare_efficiency", "seller_ask_index",
                           "avg_seller_ask", "avg_buyer_bid", "seller_high_ask_ratio", "seller_low_ask_ratio"):
                self.assertIsNone(result["metrics"][metric])
            self.assertEqual(result["metrics"]["num_trades"], 0)
            self.assertEqual(result["metrics"]["total_welfare"], 0)
            self.assertIsNone(result["usage"]["total_tokens"])
            json.dumps(result, allow_nan=False)

    def test_no_trades_have_zero_returns_and_null_average(self):
        config = make_config(sellers=1, buyers=1, rounds=1, overseer=False)
        result = summarize(config, [make_round(1, {"Seller_1": 105}, {"Buyer_1": 85})])
        self.assertEqual(result["agents"]["sellers"][0]["profit"], 0)
        self.assertEqual(result["agents"]["buyers"][0]["surplus"], 0)
        self.assertEqual(result["metrics"]["capacity_utilization"], 0)
        self.assertIsNone(result["metrics"]["avg_trade_price"])
        self.assertEqual(result["metrics"]["seller_ask_index"], 1.5)

    def test_reference_prices_and_heterogeneous_maximum_welfare(self):
        config = make_config(sellers=3, buyers=2, rounds=2, overseer=False)
        for agent, cost in zip(config["agents"]["sellers"], [70, 85, 95]):
            agent["valuation"] = cost
        for agent, value in zip(config["agents"]["buyers"], [80, 110]):
            agent["valuation"] = value
        result = summarize(config, [make_round(1), make_round(2)])
        self.assertEqual(result["metrics"]["max_possible_welfare"], 80)
        self.assertEqual(result["metrics"]["max_possible_trades"], 4)
        self.assertEqual(competitive_equilibrium([110, 80], [70, 85, 95]), (90, 1))
        self.assertEqual(jpm_price([110, 80], [70, 85, 95]), 110)
        self.assertIsNone(jpm_price([], [80]))
        self.assertIsNone(jpm_price([70], [80]))
        config["agents"]["sellers"][0]["valuation"] = 110
        config["agents"]["buyers"][0]["valuation"] = 110
        config["agents"]["sellers"] = config["agents"]["sellers"][:1]
        config["agents"]["buyers"] = config["agents"]["buyers"][:1]
        self.assertIsNone(summarize(config, [make_round(1), make_round(2)])["metrics"]["seller_ask_index"])

    def test_trade_weighted_price_and_agent_specific_returns(self):
        config = make_config(sellers=2, buyers=2, rounds=2, overseer=False)
        config["agents"]["sellers"][1]["valuation"] = 85
        config["agents"]["buyers"][1]["valuation"] = 105
        rounds = [
            make_round(1, trades=[make_trade(1, "Seller_1", "Buyer_1", 90)]),
            make_round(2, trades=[make_trade(2, "Seller_1", "Buyer_1", 96), make_trade(3, "Seller_2", "Buyer_2", 99)]),
        ]
        result = summarize(config, rounds)
        self.assertEqual(result["metrics"]["avg_trade_price"], 95)
        self.assertEqual(result["metrics"]["combined_seller_profits"], 40)
        self.assertEqual(result["metrics"]["combined_buyer_surplus"], 20)
        self.assertEqual([agent["num_trades"] for agent in result["agents"]["sellers"]], [2, 1])

    def test_warning_counts_transitions_and_final_round(self):
        config = make_config(rounds=3)
        rounds = [make_round(i + 1, overseer={"coordination_score": score,
                                             "active_before": before, "active_after": after})
                  for i, (score, before, after) in enumerate([(3.5, False, True), (3.8, True, True), (2.9, True, False)])]
        summary = summarize(config, rounds)["agents"]["overseer"]
        self.assertEqual(summary["threshold_met_round_count"], 2)
        self.assertEqual(summary["warning_round_count"], 2)
        self.assertEqual(summary["first_warning_round"], 1)
        self.assertFalse(summary["warning_active"])
        config["settings"]["rounds"] = 1
        summary = summarize(config, rounds[:1])["agents"]["overseer"]
        self.assertEqual(summary["warning_round_count"], 0)
        self.assertTrue(summary["warning_active"])

    def test_error_and_retry_usage_counts(self):
        config = make_config(rounds=0, overseer=False)
        calls = [
            {"event": "request", "call_id": "c1", "attempt": 1},
            {"event": "response", "call_id": "c1", "attempt": 1, "status": "api_error", "usage": None},
            {"event": "request", "call_id": "c1", "attempt": 2},
            {"event": "response", "call_id": "c1", "attempt": 2, "status": "parse_error", "parse_error": "bad JSON",
             "usage": {"input_tokens": 10, "output_tokens": 2}},
            {"event": "request", "call_id": "c1", "attempt": 3},
            {"event": "response", "call_id": "c1", "attempt": 3, "status": "success",
             "usage": {"input_tokens": 3, "output_tokens": None, "total_tokens": 5}},
            {"event": "request", "call_id": "c2", "attempt": 1},
            {"event": "response", "call_id": "c2", "attempt": 1, "status": "validation_error",
             "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 9}},
            {"event": "request", "call_id": "c3", "attempt": 1},
        ]
        result = summarize(config, [], calls)
        self.assertEqual(result["calls"], {"total_attempts": 5, "api_errors": 1, "parse_errors": 1, "retries": 2})
        self.assertEqual(result["usage"], {"reported_calls": 2, "input_tokens": 15, "output_tokens": 3, "total_tokens": 21})

    def test_incomplete_and_zero_usage(self):
        config = make_config(rounds=0, overseer=False)
        for usage in (None, {}, {"input_tokens": 1}, {"input_tokens": True, "output_tokens": 2}):
            with self.subTest(usage=usage):
                result = summarize(config, [], [{"event": "response", "status": "success", "usage": usage}])
                self.assertEqual(result["usage"], {"reported_calls": 0, "input_tokens": None,
                                                   "output_tokens": None, "total_tokens": None})
        result = summarize(config, [], [{"event": "response", "status": "success",
                                        "usage": {"input_tokens": 0, "output_tokens": 0}}])
        self.assertEqual(result["usage"]["reported_calls"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 0)

    def test_reject_misleading_completion_and_invalid_numbers(self):
        config = make_config(rounds=2, overseer=False)
        for rounds in ([], [make_round(1)], [make_round(1), make_round(1)], [make_round(2)],
                       [make_round(1), make_round(2), make_round(3)]):
            with self.assertRaises(ValueError):
                summarize(config, rounds)
        for value in (True, float("nan"), float("inf"), "94"):
            with self.assertRaises(ValueError):
                summarize(config, [make_round(1, {"Seller_1": value}), make_round(2)])
        running = summarize(config, [], status="running")
        self.assertIsNone(running["ended_at"])


class ReaderAndAnalysisTests(unittest.TestCase):
    def test_strict_reader_and_recompute(self):
        config, rounds, calls = example_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / config["run_id"]
            run.mkdir()
            (run / "config.json").write_text(json.dumps(config))
            (run / "market.jsonl").write_text("\n".join(map(json.dumps, rounds)))
            (run / "calls.jsonl").write_text("\n".join(map(json.dumps, calls)))
            (run / "summary.json").write_text(json.dumps(summarize(config, rounds, calls)))
            self.assertEqual(find_run_dirs(root), [run])
            self.assertEqual(find_run_dirs(run), [run])
            self.assertEqual(len(read_jsonl(run / "market.jsonl")), 25)
            self.assertEqual(verify_summary(run)["metrics"]["num_trades"], 80)
            changed = read_json(run / "summary.json")
            changed["metrics"]["num_trades"] = 79
            (run / "summary.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                verify_summary(run)
            for invalid in ('{"x": NaN}', '[]', '{"x":'):
                (run / "market.jsonl").write_text(invalid)
                with self.assertRaisesRegex(ValueError, "line 1"):
                    read_jsonl(run / "market.jsonl")

    def test_group_rejects_different_effective_configuration(self):
        from src.evaluation.analysis import group_runs
        changes = [
            lambda config: config["agents"]["overseer"].update(threshold=3.0),
            lambda config: config["agents"]["overseer"].update(model="different-jev"),
            lambda config: config["agents"]["sellers"][0].update(model="different-trader"),
            lambda config: config["agents"]["buyers"][0].update(valuation=105),
            lambda config: config["agents"]["buyers"].pop(),
            lambda config: config["settings"].update(temperature=0.5),
            lambda config: config["initialization"].update(initial_ask_range=[96, 99]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for number in range(2):
                    config = make_config()
                    if number:
                        change(config)
                    run = root / f"seed{number}"
                    run.mkdir()
                    (run / "config.json").write_text(json.dumps(config))
                    (run / "summary.json").write_text('{"status": "completed"}')
                    (run / "market.jsonl").touch()
                    (run / "calls.jsonl").touch()
                with self.assertRaisesRegex(ValueError, "Use different tags"):
                    group_runs(root)

    def test_group_allows_seed_identity_and_actual_initial_quote_changes(self):
        from src.evaluation.analysis import group_runs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(2):
                config = make_config()
                config.update(seed=number, run_id=f"run{number}", created_at=f"time{number}", result_dir=f"root{number}")
                for side, quote_key in (("sellers", "initial_ask"), ("buyers", "initial_bid")):
                    for index, agent in enumerate(config["agents"][side]):
                        agent.update(id=f"{side}_{number}_{index}", company=f"company{number}_{index}")
                        agent[quote_key] = 95 + number if side == "sellers" else 80 + number
                run = root / f"seed{number}"
                run.mkdir()
                (run / "config.json").write_text(json.dumps(config))
                (run / "summary.json").write_text('{"status": "completed"}')
                (run / "market.jsonl").touch()
                (run / "calls.jsonl").touch()
            groups = group_runs(root)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(next(iter(groups.values()))), 2)

    def test_analysis_and_plot_new_records(self):
        import matplotlib
        matplotlib.use("Agg")
        from src.evaluation.analysis import bootstrap_ci, group_runs, load_experiment_data
        from src.evaluation.plots import aggregate_round_metrics, create_market_figure
        config, rounds, calls = example_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for index in range(3):
                run = root / f"seed{index}"
                run.mkdir()
                current = copy.deepcopy(config)
                current["tag"] = "misleading_no_oversight_name"
                summary = summarize(current, rounds, calls)
                if index == 2:
                    summary["status"] = "failed"
                (run / "config.json").write_text(json.dumps(current))
                (run / "market.jsonl").write_text("\n".join(map(json.dumps, rounds)))
                (run / "calls.jsonl").write_text("")
                (run / "summary.json").write_text(json.dumps(summary))
                runs.append(run)
            groups = group_runs(root)
            self.assertEqual(len(groups), 1)
            label = next(iter(groups))
            self.assertIn("overseer=on", label)
            self.assertEqual(len(groups[label]), 2)
            self.assertEqual(load_experiment_data(root, "combined_seller_profits")[label], [1000, 1000])
            numbers, prices, margins = aggregate_round_metrics(runs[:2], "avg_trade_price", 7)
            self.assertEqual(numbers.tolist(), list(range(1, 8)))
            self.assertTrue(all(math.isnan(value) for value in prices[:5]))
            self.assertEqual(margins[5:].tolist(), [0, 0])
            self.assertEqual(prices[5:].tolist(), [95, 95])
            image = create_market_figure(root, root / "outputs")
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(bootstrap_ci([10, 10], num_replications=10), (10, 10, 10))
            with self.assertRaises(ValueError):
                bootstrap_ci([])


if __name__ == "__main__":
    unittest.main()
