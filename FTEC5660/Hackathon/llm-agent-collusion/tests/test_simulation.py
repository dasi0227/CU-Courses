import json
import re
from collections import Counter
from types import SimpleNamespace

import pytest
import httpx2
from openai import APIConnectionError
from pydantic import ValidationError
from typesafe_sdk import SystemOneResponse

from src import simulation
from src.llm import client as clients
from src.records.reader import read_json, read_jsonl
from src.evaluation.evaluate import verify_summary
from src.schemas import DEFAULT_MODEL, ExperimentParams


def params(**changes):
    values = dict(seller_valuations=[80] * 4, buyer_valuations=[100] * 4,
                  seller_models=[DEFAULT_MODEL] * 4, buyer_models=[DEFAULT_MODEL] * 4,
                  rounds=25, seed=0, tag="test", seller_messages_enabled=True,
                  overseer_enabled=True)
    return ExperimentParams(**(values | changes))


def trader_decision(agent_id, request, price=90, **changes):
    seller = agent_id.startswith("Seller")
    result = {
        "reflection": "Review previous outcomes.",
        "plan_for_this_hour": "Quote near the current market.",
        "ask" if seller else "bid": price,
        "new_memory": "Prepared this hour's quote.",
        "scratch_pad_update": "Next strategy",
    }
    message_key = "message_to_sellers" if seller else "message_to_buyers"
    if f'"{message_key}":' in request["messages"][0]["content"]:
        result[message_key] = "x" * 600
    return result | changes


@pytest.fixture
def fake_sdks(monkeypatch):
    state = SimpleNamespace(trader_requests=[], jev_requests=[], attempts=Counter(),
                            trader_hook=None, jev_hook=None, closed=0)

    class TraderSDK:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **request):
            state.trader_requests.append(request)
            system, user = [entry["content"] for entry in request["messages"]]
            agent_id = re.search(r"Your name is (.*?)\.", system).group(1)
            hour = int(re.search(r"currently at Hour (\d+)", user).group(1))
            key = (agent_id, hour)
            state.attempts[key] += 1
            ask = "Seller " in agent_id
            if state.trader_hook:
                result = state.trader_hook(agent_id, hour, state.attempts[key], request)
            else:
                price = (98 if hour <= 5 else 94 if hour <= 15 else 88) if ask else (
                    [82, 84, 86, 88, 89][hour - 1] if hour <= 5 else 96 if hour <= 15 else 92)
                result = trader_decision(agent_id, request, price)
            return SimpleNamespace(
                id=f"mock-{agent_id}-{hour}", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                    content=result if isinstance(result, str) else json.dumps(result)))],
            )

        def close(self):
            state.closed += 1

    class JevSDK:
        def __init__(self, **kwargs):
            assert kwargs["retry"].max_retries == 0

        def system_one(self, **request):
            state.jev_requests.append(request)
            hour = request["state"]["current_round"]["round"]
            score = state.jev_hook(hour, request) if state.jev_hook else (2.6 if hour in (7, 12) else 1.0)
            return SystemOneResponse.model_validate({
                "model": "mock-jev", "usage": {"input_tokens": 10, "output_tokens": 5},
                "answers": {"coordination": {"type": "score", "score": score, "confidence": 0.8,
                                                "probabilities": {}, "legend": {}}},
            })

        def close(self):
            state.closed += 1

    monkeypatch.setattr(clients, "make_client", TraderSDK)
    monkeypatch.setattr(clients, "TypeSafeClient", JevSDK)
    real_trader, real_jev = simulation.get_client, simulation.get_overseer_client

    def trader(**kwargs):
        result = real_trader(**kwargs)
        result.max_retries, result.initial_retry_delay = 1, 0
        return result

    def jev(**kwargs):
        result = real_jev(**kwargs)
        result.max_retries, result.initial_retry_delay = 1, 0
        return result

    monkeypatch.setattr(simulation, "get_client", trader)
    monkeypatch.setattr(simulation, "get_overseer_client", jev)
    return state


def load_run(root):
    run = next(root.iterdir())
    return run, read_json(run / "config.json"), read_jsonl(run / "market.jsonl"), read_jsonl(run / "calls.jsonl")


def test_full_example_and_linked_records(tmp_path, fake_sdks):
    summary = simulation.run_simulation(params(), str(tmp_path))
    run, config, market, calls = load_run(tmp_path)
    assert {path.name for path in run.iterdir()} == {"config.json", "market.jsonl", "calls.jsonl", "summary.json"}
    assert config["run_id"].endswith("_test_seed0") and config["tag"] == "test"
    assert config["agents"]["overseer"]["model"] == "jev-latest"
    assert all(agent["model"] == "deepseek/deepseek-v4-flash"
               for side in ("sellers", "buyers") for agent in config["agents"][side])
    assert verify_summary(run) == summary
    assert fake_sdks.closed == 9
    expected = dict(num_trades=80, avg_trade_price=92.5, combined_seller_profits=1000,
                    combined_buyer_surplus=600, total_welfare=1600, avg_seller_ask=92.4,
                    avg_buyer_bid=92.36, seller_ask_index=0.24, welfare_efficiency=0.8,
                    avg_seller_ask_dispersion=0, max_possible_trades=100)
    for key, value in expected.items():
        assert summary["metrics"][key] == pytest.approx(value)
    assert summary["calls"] == dict(total_attempts=225, api_errors=0, parse_errors=0, retries=0)
    assert summary["usage"] == dict(reported_calls=225, input_tokens=20250, output_tokens=10125, total_tokens=30375)
    assert summary["agents"]["overseer"]["first_warning_round"] == 7
    assert summary["agents"]["overseer"]["warning_round_count"] == 2
    assert summary["agents"]["overseer"]["threshold_met_round_count"] == 2
    assert not summary["agents"]["overseer"]["warning_active"]
    assert [m["round"] for m in market if m["overseer"]["active_before"]] == [8, 13]
    for request in fake_sdks.trader_requests:
        assert request["model"] == "deepseek/deepseek-v4-flash"
        assert request["extra_body"]["provider"] == {"only": ["baidu/fp8"], "allow_fallbacks": False}
        system, user = [entry["content"] for entry in request["messages"]]
        hour = int(re.search(r"currently at Hour (\d+)", user).group(1))
        if "Your name is Seller " in system:
            assert ("The regulatory authorities suspect" in system) == (hour in (8, 13))
        assert "**Role:" not in system + user
    for index, request in enumerate(fake_sdks.jev_requests):
        assert set(request["state"]) == {"previous_round", "current_round"}
        assert request["state"]["current_round"]["seller_messages"] == market[index]["seller_messages"]
        assert all(len(msg) == 500 for msg in market[index]["seller_messages"].values())
        assert request["state"]["previous_round"] == (None if index == 0 else {
            "round": index, "seller_messages": market[index - 1]["seller_messages"]})
    requests = {(c["call_id"], c["attempt"]): c for c in calls if c["event"] == "request"}
    responses = {(c["call_id"], c["attempt"]): c for c in calls if c["event"] == "response"}
    assert len(requests) == 225 and requests.keys() == responses.keys()
    for key, request in requests.items():
        response = responses[key]
        if request["agent_id"] == "Overseer":
            assert "messages" not in request
            assert response["finish_reason"] is None
            native = json.loads(response["raw_text"])["answers"]["coordination"]["score"]
            assert native + 1 == market[request["round"] - 1]["overseer"]["coordination_score"]
    for row in market:
        assert "time" not in row and "event" not in row
        assert all("round_number" not in trade for trade in row["trades"])
    assert len({t["trade_id"] for row in market for t in row["trades"]}) == 80


def test_retry_parse_and_api_are_real_attempts(tmp_path, fake_sdks):
    def hook(agent_id, hour, attempt, request):
        if agent_id.startswith("Seller") and attempt == 1:
            return "not json"
        if agent_id.startswith("Buyer") and attempt == 1:
            raise APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
        return trader_decision(agent_id, request)
    fake_sdks.trader_hook = hook
    summary = simulation.run_simulation(params(rounds=1, overseer_enabled=False), str(tmp_path))
    _, _, market, _ = load_run(tmp_path)
    assert summary["calls"] == dict(total_attempts=16, api_errors=4, parse_errors=4, retries=8)
    assert summary["usage"]["reported_calls"] == 12
    assert len(market[0]["trades"]) == 4


def test_failed_overseer_preserves_state_and_excludes_partial_round(tmp_path, fake_sdks):
    def score(hour, request):
        if hour == 2:
            raise RuntimeError("Jev unavailable")
        return 3
    fake_sdks.jev_hook = score
    fake_sdks.trader_hook = lambda agent_id, hour, attempt, request: trader_decision(agent_id, request)
    with pytest.raises(clients.ModelCallError):
        simulation.run_simulation(params(rounds=3), str(tmp_path))
    run, _, market, _ = load_run(tmp_path)
    summary = verify_summary(run)
    assert summary["status"] == "failed" and summary["completed_rounds"] == 1
    assert market[1]["status"] == "failed" and len(market[1]["trades"]) == 4
    assert market[1]["overseer"] == dict(coordination_score=None, active_before=True, active_after=True)
    assert summary["metrics"]["num_trades"] == 4
    assert summary["metrics"]["max_possible_trades"] == 12
    assert summary["agents"]["overseer"]["warning_active"] is True
    assert summary["agents"]["overseer"]["warning_round_count"] == 0
    assert summary["error"]["agent_id"] == "Overseer" and summary["error"]["round"] == 2


def test_failed_trader_preserves_other_decisions(tmp_path, fake_sdks):
    def hook(agent_id, hour, attempt, request):
        if agent_id.startswith("Buyer"):
            raise RuntimeError("offline")
        return trader_decision(agent_id, request, 94, message_to_sellers="abc")
    fake_sdks.trader_hook = hook
    with pytest.raises(clients.ModelCallError):
        simulation.run_simulation(params(rounds=1), str(tmp_path))
    run, _, market, _ = load_run(tmp_path)
    assert market[0]["status"] == "failed" and len(market[0]["seller_asks"]) == 4
    assert not market[0]["trades"] and not fake_sdks.jev_requests
    summary = verify_summary(run)
    assert summary["completed_rounds"] == 0 and summary["metrics"]["avg_seller_ask"] is None


def test_initial_orders_withdrawn_by_explicit_null(tmp_path, fake_sdks):
    def withdraw(agent_id, hour, attempt, request):
        result = trader_decision(agent_id, request, None)
        for key in ("message_to_sellers", "message_to_buyers"):
            if key in result:
                result[key] = None
        return result
    fake_sdks.trader_hook = withdraw
    summary = simulation.run_simulation(params(rounds=1), str(tmp_path))
    _, config, market, _ = load_run(tmp_path)
    assert market[0]["seller_asks"] == {} and market[0]["buyer_bids"] == {}
    assert sorted(market[0]["seller_withdrawals"]) == sorted(a["id"] for a in config["agents"]["sellers"])
    assert sorted(market[0]["buyer_withdrawals"]) == sorted(a["id"] for a in config["agents"]["buyers"])
    assert market[0]["remaining_order_book"]["asks"] == []
    assert market[0]["remaining_order_book"]["bids"] == []
    assert all(a["bids_by_round"] == [None] for a in summary["agents"]["buyers"])
    assert summary["metrics"]["avg_trade_price"] is None
    assert fake_sdks.jev_requests[0]["state"]["current_round"]["seller_messages"] == {}


@pytest.mark.parametrize("side", ["buyer", "seller"])
@pytest.mark.parametrize("messages_enabled", [False, True])
@pytest.mark.parametrize("defect", ["missing", "misspelled"])
def test_agent_schema_failures_retry_before_applying_decisions(tmp_path, fake_sdks, side, messages_enabled, defect):
    def hook(agent_id, hour, attempt, request):
        result = trader_decision(agent_id, request, new_memory=f"Accepted memory {hour}",
                                 scratch_pad_update=f"Accepted notes {hour}")
        message_key = "message_to_sellers" if agent_id.startswith("Seller") else "message_to_buyers"
        if message_key in result:
            result[message_key] = "Accepted message"
        if agent_id.startswith(side.title()) and hour == 1 and attempt == 1:
            result["ask" if side == "seller" else "bid"] = 1
            result["new_memory"] = "Rejected memory"
            del result["scratch_pad_update"]
            if defect == "misspelled":
                result["scratch_patch_update"] = "Rejected notes"
            if message_key in result:
                result[message_key] = "Rejected message"
        return result

    fake_sdks.trader_hook = hook
    summary = simulation.run_simulation(params(rounds=2, overseer_enabled=False,
        buyer_messages_enabled=messages_enabled, seller_messages_enabled=messages_enabled), str(tmp_path))
    run, _, market, calls = load_run(tmp_path)
    assert verify_summary(run) == summary
    assert summary["calls"] == dict(total_attempts=20, api_errors=0, parse_errors=0, retries=4)
    assert summary["completed_rounds"] == 2 and summary["metrics"]["num_trades"] == 8
    failures = [c for c in calls if c["event"] == "response" and c["status"] == "validation_error"]
    assert len(failures) == 4
    assert all("Rejected memory" in c["raw_text"] and "scratch_pad_update" in c["error"]["message"] for c in failures)
    for row in market:
        assert set(row["seller_asks"].values()) == {90.0}
        assert set(row["buyer_bids"].values()) == {90.0}
        for message_key, quote_key in (("seller_messages", "seller_asks"), ("buyer_messages", "buyer_bids")):
            expected = dict.fromkeys(row[quote_key], "Accepted message") if messages_enabled else {}
            assert row[message_key] == expected
    requests = [c for c in calls if c["event"] == "request" and c["round"] == 2]
    assert len(requests) == 8
    for request in requests:
        context = request["messages"][1]["content"]
        assert "Accepted memory 1" in context and "Accepted notes 1" in context
        assert "Rejected" not in context


@pytest.mark.parametrize("side", ["buyer", "seller"])
def test_missing_quotes_exhaust_retries_and_fail_the_round(tmp_path, fake_sdks, side):
    price_key = "ask" if side == "seller" else "bid"
    def hook(agent_id, hour, attempt, request):
        result = trader_decision(agent_id, request)
        if agent_id.startswith(side.title()):
            del result[price_key]
        return result
    fake_sdks.trader_hook = hook
    with pytest.raises(clients.ModelCallError, match=f"Missing required fields: {price_key}"):
        simulation.run_simulation(params(rounds=1), str(tmp_path))
    run, _, market, calls = load_run(tmp_path)
    summary = verify_summary(run)
    assert summary["status"] == "failed" and summary["completed_rounds"] == 0
    assert summary["calls"]["total_attempts"] == 12 and summary["calls"]["retries"] == 4
    assert market[0]["status"] == "failed" and not market[0]["trades"]
    assert market[0]["seller_asks" if side == "seller" else "buyer_bids"] == {}
    assert sum(c.get("status") == "validation_error" for c in calls) == 8
    assert not fake_sdks.jev_requests


@pytest.mark.parametrize("initial_args, enabled", [
    ([], True), (["--initial_condition", "true"], True), (["--initial_condition", "false"], False),
])
def test_cli_multiple_seeds_initial_condition_and_model_override(tmp_path, fake_sdks, monkeypatch, initial_args, enabled):
    monkeypatch.setenv("LLM_MODEL", "custom/model")
    monkeypatch.setattr("sys.argv", ["simulation", "--rounds", "1", "--seeds", "5", "6",
        *initial_args, "--overseer_threshold", "2", "--tag", "oversight", "--result_dir", str(tmp_path)])
    simulation.main()
    assert len(list(tmp_path.iterdir())) == 2
    for run in tmp_path.iterdir():
        config, summary = read_json(run / "config.json"), verify_summary(run)
        assert config["tag"] == "oversight" and config["agents"]["overseer"] is None
        assert config["settings"]["overseer_enabled"] is False
        assert config["initialization"]["initial_condition"] is enabled
        assert all(a["model"] == "custom/model" for a in config["agents"]["sellers"])
        assert all((a["initial_ask"] is not None) == enabled for a in config["agents"]["sellers"])
        assert all((a["initial_bid"] is not None) == enabled for a in config["agents"]["buyers"])
        assert summary["agents"]["overseer"] is None
    assert not fake_sdks.jev_requests
    assert all(request["model"] == "custom/model" for request in fake_sdks.trader_requests)


@pytest.mark.parametrize("changes", [dict(overseer_threshold=x) for x in (0, 5, float("nan"), float("inf"))] + [
    dict(initial_ask_range=(100, 95)), dict(initial_bid_range=(80, float("inf"))), dict(tag="../outside")])
def test_invalid_parameters(changes):
    with pytest.raises(ValidationError):
        params(**changes)


def test_initial_ranges_reproducible_and_run_not_overwritten(tmp_path, fake_sdks):
    first = simulation.run_simulation(params(rounds=0, initial_ask_range=(97, 97), initial_bid_range=(82, 82)), str(tmp_path))
    run, config, _, _ = load_run(tmp_path)
    assert all(a["initial_ask"] == 97 for a in config["agents"]["sellers"])
    assert all(a["initial_bid"] == 82 for a in config["agents"]["buyers"])
    with pytest.raises(FileExistsError):
        # Force the same timestamp to exercise collision handling deterministically.
        from unittest.mock import patch
        with patch.object(simulation, "timestamp", return_value=config["created_at"]):
            simulation.run_simulation(params(rounds=0), str(tmp_path))
    assert read_json(run / "summary.json") == first


def test_exact_threshold_switches_next_round_and_final_trigger(tmp_path, fake_sdks):
    fake_sdks.jev_hook = lambda hour, request: [2.5, 2.49, 2.5][hour - 1]
    summary = simulation.run_simulation(params(rounds=3), str(tmp_path))
    _, _, market, _ = load_run(tmp_path)
    assert [row["overseer"]["active_before"] for row in market] == [False, True, False]
    assert [row["overseer"]["active_after"] for row in market] == [True, False, True]
    assert summary["agents"]["overseer"]["threshold_met_round_count"] == 2
    assert summary["agents"]["overseer"]["warning_round_count"] == 1
    assert summary["agents"]["overseer"]["warning_active"] is True


@pytest.mark.parametrize("arguments", [
    ["--overseer_threshold", "nan"], ["--overseer_threshold", "inf"], ["--overseer_threshold", "0.9"],
    ["--initial_bid_range", "90", "80"], ["--seeds", "1", "1"], ["--overseer_enabled"],
    ["--sellers", "-1"], ["--sellers", "21"], ["--seeds", "-1"], ["--initial_seller_ask_center", "97"],
    ["--overseer_model", "other-model"], ["--no_initial_condition"],
    ["--initial_condition"], ["--initial_condition", "yes"],
])
def test_bad_cli_fails_before_clients(arguments, fake_sdks, monkeypatch):
    monkeypatch.setattr("sys.argv", ["simulation", *arguments])
    with pytest.raises(SystemExit) as exc:
        simulation.main()
    assert exc.value.code == 2 and not fake_sdks.trader_requests and not fake_sdks.jev_requests


def test_custom_ranges_and_model_roster_through_cli(tmp_path, fake_sdks, monkeypatch):
    monkeypatch.setattr("sys.argv", ["simulation", "--rounds", "0", "--sellers", "2", "--buyers", "1",
        "--seller_models", "gpt-4.1", "gpt-4.1-mini", "--buyer_models", "test/buyer",
        "--initial_ask_range", "95", "100", "--initial_bid_range", "80", "85", "--result_dir", str(tmp_path)])
    simulation.main()
    _, config, _, _ = load_run(tmp_path)
    assert [a["model"] for a in config["agents"]["sellers"]] == ["openai/gpt-4.1", "openai/gpt-4.1-mini"]
    assert config["agents"]["buyers"][0]["model"] == "test/buyer"
    assert all(95 <= a["initial_ask"] <= 100 and round(a["initial_ask"], 2) == a["initial_ask"]
               for a in config["agents"]["sellers"])
    assert 80 <= config["agents"]["buyers"][0]["initial_bid"] <= 85


def test_zero_message_cap_means_empty_communication(tmp_path, fake_sdks):
    simulation.run_simulation(params(rounds=1, max_messages_length=0, buyer_messages_enabled=True), str(tmp_path))
    _, _, market, _ = load_run(tmp_path)
    assert market[0]["seller_messages"] == {} and market[0]["buyer_messages"] == {}
    assert fake_sdks.jev_requests[0]["state"]["current_round"]["seller_messages"] == {}


def test_client_initialization_failure_is_recorded(tmp_path, fake_sdks, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Missing test credential")
    monkeypatch.setattr(simulation, "get_overseer_client", fail)
    with pytest.raises(RuntimeError):
        simulation.run_simulation(params(rounds=1), str(tmp_path))
    run, _, market, calls = load_run(tmp_path)
    summary = verify_summary(run)
    assert summary["status"] == "failed" and summary["completed_rounds"] == 0
    assert summary["error"]["agent_id"] == "Overseer" and summary["error"]["round"] is None
    assert calls == [] and market == [] and fake_sdks.closed == 8
