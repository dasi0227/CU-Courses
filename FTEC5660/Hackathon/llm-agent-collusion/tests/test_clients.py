import json
from types import SimpleNamespace

import httpx2
from openai import APIConnectionError, OpenAI
import pytest
from typesafe_sdk import SystemOneResponse, TypeSafeClient

import src.llm.client as client_module
from src.llm.client import JevClient, ModelCallError, ModelWrapper, OpenRouterClient
from src.market.agents import COORDINATION_QUESTION, MessageOverseer, validate_decision


class Recorder:
    def __init__(self):
        self.records = []
        self.count = 0

    def new_call_id(self):
        self.count += 1
        return f"c{self.count}"

    def log_call(self, record):
        self.records.append(json.loads(json.dumps(record)))


class FakeTrader(ModelWrapper):
    def __init__(self, replies, **kwargs):
        super().__init__("mock-trader", initial_retry_delay=0, **kwargs)
        self.replies = iter(replies)

    def _request(self, messages):
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return response


def trader_response(text, usage=None, finish_reason="stop"):
    return SimpleNamespace(
        id="completion-1", _request_id="request-1", usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
    )


def decision(side="seller", messages_enabled=False, **changes):
    data = {
        "reflection": "Review previous trades.",
        "plan_for_this_hour": "Quote near the current market.",
        "ask" if side == "seller" else "bid": 94.0,
        "new_memory": "Prepared this hour's quote.",
        "scratch_pad_update": "Keep watching the market.",
    }
    if messages_enabled:
        data[f"message_to_{side}s"] = None
    return data | changes


def jev_body(score=2.6, usage=None):
    return {
        "model": "jev-test", "usage": usage or {"input_tokens": 12, "output_tokens": 4},
        "answers": {"coordination": {
            "type": "score", "score": score, "confidence": 0.6,
            "probabilities": {"0": 0.0, "1": 0.0, "2": 0.4, "3": 0.6},
            "legend": {"0": "none", "1": "ambiguous", "2": "proposal", "3": "agreement"},
        }},
    }


def make_jev(monkeypatch, handler, max_retries=0):
    monkeypatch.setattr(client_module, "load_environment", lambda: None)
    real_factory = TypeSafeClient
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return real_factory(api_key="test-key", transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(client_module, "TypeSafeClient", factory)
    return JevClient(max_retries=max_retries, initial_retry_delay=0), created


def test_trader_preserves_raw_text_and_exact_metadata():
    recorder = Recorder()
    raw = '<think>test reasoning</think>\n```json\n{"ask": 94}\n```'
    trader = FakeTrader([trader_response(raw, {"prompt_tokens": 11, "completion_tokens": 3})])
    result = trader.generate_json([{"role": "user", "content": "choose"}], round_num=2,
                                  agent_id="Seller_1", logger=recorder)
    assert result == {"ask": 94}
    request, response = recorder.records
    assert request["parameters"] == {"temperature": 0.7, "max_tokens": 8192, "top_p": 1.0}
    assert request["round"] == 2 and request["agent_id"] == "Seller_1"
    assert response["raw_text"] == raw
    assert response["request_id"] == "request-1"
    assert response["finish_reason"] == "stop"
    assert response["usage"] == {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
    assert response["parse_error"] is None


@pytest.mark.parametrize("usage, expected", [
    (None, {"input_tokens": None, "output_tokens": None, "total_tokens": None}),
    ({"prompt_tokens": 5}, {"input_tokens": 5, "output_tokens": None, "total_tokens": None}),
    ({"prompt_tokens": 0, "completion_tokens": 0}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    ({"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 10},
     {"input_tokens": 5, "output_tokens": 3, "total_tokens": 10}),
])
def test_actual_usage_not_estimated(usage, expected):
    recorder = Recorder()
    FakeTrader([trader_response("{}", usage, finish_reason=None)]).generate_json(
        [], round_num=1, agent_id="Buyer_1", logger=recorder,
    )
    assert recorder.records[1]["usage"] == expected
    assert recorder.records[1]["finish_reason"] is None


def test_api_parse_and_validation_failures_are_separate_logged_attempts():
    recorder = Recorder()
    valid = decision(ask=None)
    trader = FakeTrader([
        APIConnectionError(request=httpx2.Request("POST", "https://example.invalid")),
        trader_response("not JSON", {"prompt_tokens": 5, "completion_tokens": 2}),
        trader_response(json.dumps(decision(ask="null"))),
        trader_response(json.dumps(valid)),
    ])
    assert trader.generate_json([], round_num=1, agent_id="Seller_1", logger=recorder,
                                validator=lambda data: validate_decision(data, "seller")) == valid
    responses = recorder.records[1::2]
    assert [r["status"] for r in responses] == ["api_error", "parse_error", "validation_error", "success"]
    assert [r["attempt"] for r in responses] == [1, 2, 3, 4]
    assert {r["call_id"] for r in recorder.records} == {"c1"}
    assert responses[1]["usage"]["total_tokens"] == 7
    assert responses[1]["parse_error"] is not None
    assert responses[2]["parse_error"] is None


def test_exhausted_error_keeps_agent_context_and_redacts_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    recorder = Recorder()
    trader = FakeTrader([RuntimeError("bad secret-test-key")], max_retries=0)
    with pytest.raises(ModelCallError) as caught:
        trader.generate_json([], round_num=3, agent_id="Buyer_2", logger=recorder)
    assert caught.value.round_num == 3 and caught.value.agent_id == "Buyer_2"
    assert "secret-test-key" not in str(caught.value)
    assert "secret-test-key" not in json.dumps(recorder.records)


@pytest.mark.parametrize("price", [-1, True, "94", "null", float("inf"), float("nan")])
def test_invalid_quote_is_validation_failure(price):
    with pytest.raises(ValueError, match="ask must be"):
        validate_decision(decision(ask=price), "seller")


@pytest.mark.parametrize("side", ["buyer", "seller"])
@pytest.mark.parametrize("messages_enabled", [False, True])
def test_complete_decisions_and_explicit_null_are_valid(side, messages_enabled):
    valid = decision(side, messages_enabled)
    validate_decision(valid, side, messages_enabled)
    valid["ask" if side == "seller" else "bid"] = None
    validate_decision(valid, side, messages_enabled)
    if messages_enabled:
        valid[f"message_to_{side}s"] = "Market looks stable."
        validate_decision(valid, side, messages_enabled)


@pytest.mark.parametrize("side", ["buyer", "seller"])
@pytest.mark.parametrize("messages_enabled", [False, True])
@pytest.mark.parametrize("defect", ["missing", "misspelled"])
def test_every_missing_or_misspelled_field_retries(side, messages_enabled, defect):
    valid = decision(side, messages_enabled)
    for field in valid:
        invalid = dict(valid)
        value = invalid.pop(field)
        if defect == "misspelled":
            invalid[field + "_typo"] = value
        raw = json.dumps(invalid)
        recorder = Recorder()
        trader = FakeTrader([trader_response(raw), trader_response(json.dumps(valid))], max_retries=1)
        result = trader.generate_json(
            [], round_num=3, agent_id=f"{side.title()}_1", logger=recorder,
            validator=lambda data: validate_decision(data, side, messages_enabled),
        )
        assert result == valid
        failed, success = recorder.records[1::2]
        assert failed["status"] == "validation_error" and success["status"] == "success"
        assert field in failed["error"]["message"]
        assert failed["raw_text"] == raw and failed["parse_error"] is None
        assert [r["attempt"] for r in recorder.records[::2]] == [1, 2]
        assert {r["call_id"] for r in recorder.records} == {"c1"}


@pytest.mark.parametrize("side", ["buyer", "seller"])
@pytest.mark.parametrize("messages_enabled", [False, True])
def test_unexpected_fields_are_rejected_even_with_all_required_fields(side, messages_enabled):
    other_side = "seller" if side == "buyer" else "buyer"
    extras = ["scratch_patch_update", "plan_for_message", "unknown_field",
              "ask" if side == "buyer" else "bid", f"message_to_{other_side}s"]
    if not messages_enabled:
        extras.append(f"message_to_{side}s")
    for field in extras:
        with pytest.raises(ValueError, match=f"Unexpected fields: {field}"):
            validate_decision(decision(side, messages_enabled, **{field: None}), side, messages_enabled)


@pytest.mark.parametrize("field", ["reflection", "plan_for_this_hour", "new_memory", "scratch_pad_update"])
@pytest.mark.parametrize("value", [None, 1, [], {}])
def test_required_text_fields_must_be_strings(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a string"):
        validate_decision(decision(**{field: value}), "seller")


@pytest.mark.parametrize("side", ["buyer", "seller"])
def test_chat_message_must_be_text_or_json_null(side):
    field = f"message_to_{side}s"
    with pytest.raises(ValueError, match=f"{field} must be a string or JSON null"):
        validate_decision(decision(side, True, **{field: 42}), side, True)


def test_missing_fields_exhaust_existing_retry_limit():
    invalid = decision()
    del invalid["scratch_pad_update"]
    raw = json.dumps(invalid)
    recorder = Recorder()
    trader = FakeTrader([trader_response(raw) for _ in range(4)])
    with pytest.raises(ModelCallError, match="Missing required fields: scratch_pad_update") as caught:
        trader.generate_json([], round_num=2, agent_id="Seller_1", logger=recorder,
                             validator=lambda data: validate_decision(data, "seller"))
    assert caught.value.round_num == 2 and caught.value.agent_id == "Seller_1"
    assert [r["attempt"] for r in recorder.records[::2]] == [1, 2, 3, 4]
    assert all(r["status"] == "validation_error" and r["raw_text"] == raw for r in recorder.records[1::2])


def test_openrouter_disables_hidden_retries_and_passes_actual_model(monkeypatch):
    created = {}
    monkeypatch.setattr(client_module, "resolve_model", lambda model: "resolved-trader")
    def factory(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(client_module, "make_client", factory)
    client = OpenRouterClient("alias", seed=17)
    assert client.model_name == "resolved-trader"
    assert created["max_retries"] == 0
    assert client.parameters["seed"] == 17
    assert "extra_body" not in client.parameters
    client.close()


@pytest.mark.parametrize("succeeds", [True, False])
def test_deepseek_pins_endpoint_on_every_attempt_and_logs_routing(monkeypatch, succeeds):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if not succeeds or len(requests) < 4:
            return httpx2.Response(503, json={"error": {"message": "Provider unavailable"}})
        return httpx2.Response(200, json={
            "id": "completion-1", "object": "chat.completion", "created": 0,
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"ask": 94}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        })

    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return OpenAI(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                      http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(client_module, "make_client", factory)
    monkeypatch.setenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
    extra_body = {"reasoning": {"enabled": False}, "provider": {
        "only": ["other-provider"], "allow_fallbacks": True, "data_collection": "deny",
    }}
    original = json.loads(json.dumps(extra_body))
    trader = client_module.get_client("gpt-4.1", temperature=0.7,
                                      initial_retry_delay=0, extra_body=extra_body)
    recorder = Recorder()
    try:
        if succeeds:
            assert trader.generate_json([], round_num=1, agent_id="Seller_1", logger=recorder) == {"ask": 94}
        else:
            with pytest.raises(ModelCallError, match="Provider unavailable"):
                trader.generate_json([], round_num=1, agent_id="Seller_1", logger=recorder)
    finally:
        trader.close()
    assert len(requests) == 4 and extra_body == original
    expected_provider = {"only": ["baidu/fp8"], "allow_fallbacks": False, "data_collection": "deny"}
    for request, record in zip(requests, recorder.records[::2]):
        assert request["model"] == record["model"] == "deepseek/deepseek-v4-flash"
        assert request["provider"] == record["parameters"]["extra_body"]["provider"] == expected_provider
        assert request["reasoning"] == {"enabled": False}
        assert request["response_format"] == {"type": "json_object"}
    assert {record["call_id"] for record in recorder.records} == {"c1"}
    assert [record["attempt"] for record in recorder.records[::2]] == [1, 2, 3, 4]
    assert [record["status"] for record in recorder.records[1::2]] == (
        ["api_error"] * 3 + ["success"] if succeeds else ["api_error"] * 4
    )


def test_jev_real_sdk_state_raw_response_and_native_score(monkeypatch):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=jev_body(), headers={"x-typesafe-request-id": "jev-request"})
    client, created = make_jev(monkeypatch, handler)
    recorder = Recorder()
    overseer = MessageOverseer(client=client)
    # Recorder is deliberately not a Pydantic ExperimentLogger; inject after construction.
    object.__setattr__(overseer, "logger", recorder)
    assert overseer.run_overseer_prompt(1, {}) == 3.6
    assert overseer.run_overseer_prompt(2, {"S2": "Agreed"}, {"S1": "Keep 94"}) == 3.6
    assert requests[0]["state"] == {"previous_round": None, "current_round": {"round": 1, "seller_messages": {}}}
    assert requests[1]["state"]["previous_round"] == {"round": 1, "seller_messages": {"S1": "Keep 94"}}
    assert requests[1]["questions"]["coordination"] == COORDINATION_QUESTION.model_dump(mode="json")
    response = recorder.records[1]
    assert json.loads(response["raw_text"])["answers"]["coordination"]["score"] == 2.6
    assert response["request_id"] == "jev-request"
    assert response["finish_reason"] is None
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
    assert created[0]["retry"].max_retries == 0
    assert created[0]["model"] == requests[0]["model"] == recorder.records[0]["model"] == "jev-latest"
    assert recorder.records[0]["parameters"] == {}
    assert "messages" not in recorder.records[0]
    client.close()


def test_jev_api_failure_retries_once_per_logged_attempt(monkeypatch):
    attempts = []
    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx2.Response(503, json={"error": "busy"})
        return httpx2.Response(200, json=jev_body(score=0))
    client, _ = make_jev(monkeypatch, handler, max_retries=1)
    recorder = Recorder()
    assert client.score({}, COORDINATION_QUESTION, round_num=1, logger=recorder) == 1
    assert len(attempts) == 2
    assert [r["status"] for r in recorder.records[1::2]] == ["api_error", "success"]
    client.close()


@pytest.mark.parametrize("score", [-0.1, 3.1, True, "2", None])
def test_jev_invalid_score_fails_without_fabricating_low_score(monkeypatch, score):
    client, _ = make_jev(monkeypatch, lambda request: httpx2.Response(200, json=jev_body(score=score)))
    recorder = Recorder()
    with pytest.raises(ModelCallError):
        client.score({}, COORDINATION_QUESTION, round_num=5, logger=recorder)
    assert recorder.records[1]["status"] == "validation_error"
    assert recorder.records[1]["parse_error"] is None
    assert recorder.records[1]["usage"]["total_tokens"] == 16
    client.close()


def test_jev_malformed_json_is_parse_error(monkeypatch):
    client, _ = make_jev(monkeypatch, lambda request: httpx2.Response(200, text="not json"))
    recorder = Recorder()
    with pytest.raises(ModelCallError):
        client.score({}, COORDINATION_QUESTION, round_num=1, logger=recorder)
    assert recorder.records[1]["status"] == "parse_error"
    assert recorder.records[1]["parse_error"] is not None
    client.close()


def test_jev_sdk_response_without_http_metadata_preserves_missing_id(monkeypatch):
    monkeypatch.setattr(client_module, "load_environment", lambda: None)
    body = jev_body(3)
    monkeypatch.setattr(client_module, "TypeSafeClient", lambda **kwargs: SimpleNamespace(
        system_one=lambda **kwargs: SystemOneResponse.model_validate_json(json.dumps(body)), close=lambda: None,
    ))
    recorder = Recorder()
    client = JevClient(max_retries=0)
    assert client.score({}, COORDINATION_QUESTION, round_num=1, logger=recorder) == 4
    assert recorder.records[1]["request_id"] is None
