"""Offline contract and policy checks; all Jev responses are scripted."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from src.experiments.jev import (
    JevClient,
    JevDisabledError,
    JevError,
    ReadSelector,
)


MODEL = "jev-1.13.0"
QUESTIONS = {"relevant": {"type": "noul", "instructions": "Is the entry relevant?"}}
PLAYBOOK = (
    "## FORMULAS AND CALCULATIONS\n"
    "[calc-00001] helpful=2 harmful=0 :: Current ratio = assets / liabilities.\n"
    "[calc-00002] helpful=1 harmful=0 :: Margin = income / revenue.\n"
    "\n## COMMON MISTAKES TO AVOID\n"
    "[err-00003] helpful=0 harmful=1 :: Convert millions and billions first.\n"
)


def response(scores, **extra):
    return {"model": MODEL,
            "answers": {name: {"type": "noul", "noul": value} for name, value in scores.items()},
            "usage": {"input_tokens": 123, "output_tokens": 7}, **extra}


class JevClientTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TEST_JEV_KEY": "unit-test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def client(self, transport, **kwargs):
        return JevClient(api_key_env="TEST_JEV_KEY", enabled=True,
                         transport=transport, retry_backoff=0, **kwargs)

    def decide(self, client):
        return client.decide({"question": "Current ratio?"}, QUESTIONS,
                             call_id="sample-1", stage="test")

    def test_disabled_constructor_and_decision_do_not_call_transport(self):
        transport = Mock(side_effect=AssertionError("Network must stay disabled"))
        client = JevClient(transport=transport)
        self.assertFalse(client.enabled)
        with self.assertRaises(JevDisabledError):
            self.decide(client)
        transport.assert_not_called()
        self.assertEqual(client.events[-1]["status"], "disabled")

    def test_official_request_and_response_contract(self):
        captured = []

        def transport(request, timeout):
            captured.append((request, timeout))
            return response({"relevant": 0.875})

        client = self.client(transport, timeout=4)
        self.assertEqual(self.decide(client), {"relevant": 0.875})
        request, timeout = captured[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["questions"], QUESTIONS)
        self.assertNotIn("Authorization", payload)
        self.assertEqual(timeout, 4)
        event = client.events[-1]
        self.assertEqual(event["response_model"], MODEL)
        self.assertEqual(event["usage"]["input_tokens"], 123)
        self.assertGreaterEqual(event["latency_seconds"], 0)
        self.assertNotIn("cost_usd", event)

    def test_missing_key_never_calls_transport(self):
        transport = Mock()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(JevError, "Missing API key"):
                self.decide(self.client(transport))
        transport.assert_not_called()

    def test_version_url_timeout_and_attempts_are_validated(self):
        for options in ({"model": "jev-latest"}, {"model": "jev-preview"},
                        {"base_url": "http://api.typesafe.ai/v1"},
                        {"base_url": "https://secret@api.typesafe.ai/v1"},
                        {"max_attempts": 3}, {"max_attempts": True}, {"max_attempts": 1.0},
                        {"timeout": 0}, {"timeout": float("inf")},
                        {"retry_backoff": 4}, {"enabled": "true"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                JevClient(**options)

    def test_bad_responses_fail_instead_of_inventing_or_clipping_scores(self):
        bad = [response({}), response({"relevant": -0.1}), response({"relevant": 1.01}),
               response({"relevant": True}), response({"relevant": "0.9"}),
               response({"relevant": float("nan")}),
               response({"relevant": 0.8}, model="jev-1.14.0"),
               {"model": MODEL, "answers": {"relevant": {"type": "score", "score": 0.8}}}]
        for bad_response in bad:
            with self.subTest(response=bad_response):
                transport = Mock(return_value=bad_response)
                client = self.client(transport)
                with self.assertRaises(JevError):
                    self.decide(client)
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(client.events[-1]["status"], "failed")

    def test_transient_error_retries_once_and_records_both_attempts(self):
        transport = Mock(side_effect=[URLError("connection lost"), response({"relevant": 0.7})])
        client = self.client(transport)
        self.assertEqual(self.decide(client), {"relevant": 0.7})
        self.assertEqual(transport.call_count, 2)
        self.assertEqual([event["status"] for event in client.events], ["failed", "success"])
        self.assertTrue(client.events[0]["will_retry"])
        self.assertEqual(client.events[1]["attempt"], 2)

    def test_timeout_has_at_most_two_attempts_and_no_fallback_score(self):
        transport = Mock(side_effect=TimeoutError("timed out"))
        client = self.client(transport)
        with self.assertRaises(JevError):
            self.decide(client)
        self.assertEqual(transport.call_count, 2)
        self.assertFalse(client.events[-1]["will_retry"])

    def test_auth_error_is_not_retried(self):
        transport = Mock(side_effect=HTTPError("https://api.typesafe.ai", 401, "Unauthorized", {}, None))
        with self.assertRaises(JevError):
            self.decide(self.client(transport))
        self.assertEqual(transport.call_count, 1)

    def test_audit_never_contains_key_even_when_error_echoes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "decisions.jsonl"
            transport = Mock(side_effect=ValueError("bad unit-test-secret"))
            client = self.client(transport, audit_path=audit)
            with self.assertRaises(JevError) as failure:
                self.decide(client)
            self.assertNotIn("unit-test-secret", str(failure.exception))
            self.assertNotIn("unit-test-secret", audit.read_text())
            self.assertNotIn("unit-test-secret", json.dumps(client.events))
            self.assertIn("[REDACTED]", audit.read_text())

    def test_audit_redacts_credentials_before_json_escaping(self):
        secret = 'a-key-with-"quotes"-and-\\-slashes'
        with patch.dict(os.environ, {"TEST_JEV_KEY": secret}):
            client = self.client(Mock())
            event = client.record_event({"error": secret, "nested": {"message": secret}})
            self.assertEqual(event["error"], "[REDACTED]")
            self.assertEqual(event["nested"]["message"], "[REDACTED]")


class ReadSelectorTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TEST_JEV_KEY": "offline-test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_batches_ranks_and_preserves_original_lines_and_headings(self):
        captured = []

        def transport(request, timeout):
            payload = json.loads(request.data)
            captured.append(payload)
            self.assertEqual(set(payload["state"]), {"question", "candidates"})
            for index, question in enumerate(payload["questions"].values()):
                self.assertEqual(question["instructions"], (
                    f"Does candidates[{index}] contain a formula, financial concept, "
                    "unit conversion, or solving strategy relevant to the question? "
                    "Judge relevance only; do not solve the question or verify arithmetic."
                ))
                self.assertEqual(question["criteria"], {
                    "true": "Relevant and potentially usable for this question",
                    "false": "Unrelated to solving this question",
                })
            by_id = {"calc-00001": 0.95, "calc-00002": 0.1, "err-00003": 0.8}
            scores = {name: by_id[candidate["id"]]
                      for name, candidate in zip(payload["questions"], payload["state"]["candidates"])}
            return response(scores)

        client = JevClient(enabled=True, api_key_env="TEST_JEV_KEY", transport=transport)
        selected, metadata = ReadSelector(client, k=2, batch_size=2).select(
            PLAYBOOK, "What is the current ratio?", call_id="test-4", stage="test")
        self.assertEqual([len(item["state"]["candidates"]) for item in captured], [2, 1])
        self.assertIn("## FORMULAS AND CALCULATIONS\n", selected)
        self.assertIn("## COMMON MISTAKES TO AVOID\n", selected)
        self.assertIn(PLAYBOOK.splitlines()[1] + "\n", selected)
        self.assertIn(PLAYBOOK.splitlines()[-1] + "\n", selected)
        self.assertNotIn("[calc-00002]", selected)
        self.assertEqual(metadata["selected_ids"], ["calc-00001", "err-00003"])
        self.assertEqual(metadata["jev_calls"], 2)
        self.assertEqual(client.events[-1]["event"], "read_selection")

    def test_ties_preserve_original_order(self):
        def transport(request, timeout):
            questions = json.loads(request.data)["questions"]
            return response({name: 0.5 for name in questions})

        client = JevClient(enabled=True, api_key_env="TEST_JEV_KEY", transport=transport)
        selected, metadata = ReadSelector(client, 1).select(
            PLAYBOOK, "A question", call_id="tie", stage="val")
        self.assertEqual(metadata["selected_ids"], ["calc-00001"])
        self.assertNotIn("[calc-00002]", selected)

    def test_multiline_entries_are_selected_or_removed_as_a_whole(self):
        playbook = (
            "## FORMULAS\n"
            "[calc-00001] helpful=0 harmful=0 :: First formula.\n"
            "  Explanation of first formula.\n"
            "[calc-00002] helpful=0 harmful=0 :: Second formula.\n"
            "  Explanation of second formula.\n"
            "## OTHERS\n"
        )
        client = JevClient(enabled=True, api_key_env="TEST_JEV_KEY",
                           transport=lambda req, timeout: response({"relevance_0": 0.1, "relevance_1": 0.9}))
        selected, _ = ReadSelector(client, 1).select(playbook, "Question", call_id="multiline", stage="test")
        self.assertNotIn("first formula", selected)
        self.assertIn("Second formula.\n  Explanation of second formula.\n", selected)
        self.assertTrue(selected.endswith("## OTHERS\n"))

    def test_empty_and_small_books_do_not_request_unneeded_decisions(self):
        transport = Mock(side_effect=AssertionError("No request needed"))
        client = JevClient(transport=transport)
        for book in ("## OTHERS\n", PLAYBOOK):
            selected, metadata = ReadSelector(client, 5).select(
                book, "Question", call_id="small", stage="test")
            self.assertEqual(selected, book)
            self.assertEqual(metadata["jev_calls"], 0)
        transport.assert_not_called()

    def test_decision_failure_propagates(self):
        selector = ReadSelector(JevClient(), 1)
        with self.assertRaises(JevDisabledError):
            selector.select(PLAYBOOK, "Question", call_id="disabled", stage="test")

    def test_invalid_sizes_and_duplicate_ids_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ReadSelector(JevClient(), value)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ReadSelector(JevClient(), 1).select(
                PLAYBOOK + PLAYBOOK, "Question", call_id="duplicate", stage="test")

if __name__ == "__main__":
    unittest.main()
