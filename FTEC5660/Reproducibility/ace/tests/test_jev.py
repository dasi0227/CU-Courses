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
CRITERIA = [
    "Unrelated to solving this question.",
    "Only topically related or a generic reminder; provides no usable solving step.",
    "Usable for a supporting step, but does not directly support the core formula or a key calculation.",
    "Directly supports the core formula or a key calculation step for this question.",
]
QUESTIONS = {"relevant": {"type": "score", "instructions": "Rate the entry's usefulness.",
                          "criteria": CRITERIA}}
PLAYBOOK = (
    "## FORMULAS AND CALCULATIONS\n"
    "[calc-00001] helpful=2 harmful=0 :: Current ratio = assets / liabilities.\n"
    "[calc-00002] helpful=1 harmful=0 :: Margin = income / revenue.\n"
    "\n## COMMON MISTAKES TO AVOID\n"
    "[err-00003] helpful=0 harmful=1 :: Convert millions and billions first.\n"
)


def response(scores, confidences=None, **extra):
    confidences = confidences or {}
    return {"model": MODEL,
            "answers": {name: {"type": "score", "score": value,
                                "confidence": confidences.get(name, 0.8)}
                        for name, value in scores.items()},
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
            return response({"relevant": 2.875}, {"relevant": 0.25})

        client = self.client(transport, timeout=4)
        self.assertEqual(self.decide(client), {
            "relevant": {"type": "score", "score": 2.875, "confidence": 0.25}})
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
        self.assertEqual(event["scores"], {"relevant": 2.875})
        self.assertEqual(event["confidences"], {"relevant": 0.25})
        self.assertEqual(event["answers"]["relevant"], {
            "type": "score", "score": 2.875, "confidence": 0.25})
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

    def test_fractional_score_and_optional_api_details_are_preserved(self):
        payload = response({"relevant": 2.625}, {"relevant": 0.1})
        payload["answers"]["relevant"].update({
            "probabilities": {str(index): value for index, value in enumerate((0.0, 0.0, 0.375, 0.625))},
            "legend": {str(index): level for index, level in enumerate(CRITERIA)},
        })
        client = self.client(Mock(return_value=payload))
        self.assertEqual(self.decide(client), payload["answers"])
        self.assertEqual(client.events[-1]["answers"], payload["answers"])
        self.assertEqual(client.events[-1]["scores"]["relevant"], 2.625)

    def test_malformed_optional_distribution_and_legend_are_rejected(self):
        bad_details = [
            {"probabilities": [0, 0, 0, 1]},
            {"probabilities": {"0": 0.5, "1": 0.5}},
            {"probabilities": {"0": False, "1": 0, "2": 0, "3": 1}},
            {"probabilities": {"0": "0", "1": 0, "2": 0, "3": 1}},
            {"probabilities": {"0": -0.01, "1": 0, "2": 0, "3": 1}},
            {"probabilities": {"0": 1.01, "1": 0, "2": 0, "3": 1}},
            {"probabilities": {"0": float("nan"), "1": 0, "2": 0, "3": 1}},
            {"probabilities": {"0": float("inf"), "1": 0, "2": 0, "3": 1}},
            {"legend": CRITERIA},
            {"legend": {"0": "Only one level"}},
            {"legend": {str(index): index for index in range(4)}},
        ]
        for details in bad_details:
            with self.subTest(details=details):
                payload = response({"relevant": 2.625})
                payload["answers"]["relevant"].update(details)
                transport = Mock(return_value=payload)
                client = self.client(transport)
                with self.assertRaises(JevError):
                    self.decide(client)
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(client.events[-1]["status"], "failed")

    def test_optional_probabilities_total_is_not_validated_or_normalized(self):
        distributions = [
            {"0": 0, "1": 0.0, "2": 0.49, "3": 0.5},
            {"0": 0, "1": 0.0, "2": 0.51, "3": 0.5},
            {"0": 0, "1": 0.0, "2": 0, "3": 0},
            {"0": 1, "1": 1.0, "2": 1, "3": 1},
        ]
        for probabilities in distributions:
            with self.subTest(probabilities=probabilities), tempfile.TemporaryDirectory() as directory:
                payload = response({"relevant": 2.625}, {"relevant": 0.125})
                payload["answers"]["relevant"]["probabilities"] = probabilities
                transport = Mock(return_value=payload)
                audit = Path(directory) / "jev_audit.jsonl"
                client = self.client(transport, audit_path=audit)
                answers = self.decide(client)
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(answers["relevant"]["score"], 2.625)
                self.assertEqual(answers["relevant"]["confidence"], 0.125)
                logged = json.loads(audit.read_text())["answers"]["relevant"]["probabilities"]
                for values in (answers["relevant"]["probabilities"], logged):
                    self.assertEqual(values, probabilities)
                    for key, value in probabilities.items():
                        self.assertIs(type(values[key]), type(value))
                self.assertEqual(client.events[-1]["status"], "success")

    def test_grade_endpoints_and_audit_file_keep_score_and_confidence(self):
        for value, confidence in ((0, 0), (3, 1)):
            with self.subTest(score=value), tempfile.TemporaryDirectory() as directory:
                audit = Path(directory) / "jev_audit.jsonl"
                client = self.client(Mock(return_value=response(
                    {"relevant": value}, {"relevant": confidence})), audit_path=audit)
                self.decide(client)
                event = json.loads(audit.read_text())
                self.assertEqual(event["answers"]["relevant"]["score"], value)
                self.assertEqual(event["scores"]["relevant"], value)
                self.assertEqual(event["confidences"]["relevant"], confidence)

    def test_bad_responses_fail_instead_of_inventing_or_clipping_scores(self):
        bad = [response({}), response({"relevant": -0.1}), response({"relevant": 3.01}),
               response({"relevant": True}), response({"relevant": "2.9"}),
               response({"relevant": float("nan")}),
               response({"relevant": float("inf")}),
               response({"relevant": 2.8}, model="jev-1.14.0"),
               response({"relevant": 2.8, "unexpected": 1.0}),
               {"model": MODEL, "answers": {"relevant": {"type": "noul", "noul": 0.8}}}]
        for value in (-0.01, 1.01, True, "0.8", float("nan"), float("inf"), None):
            bad.append(response({"relevant": 2.8}, {"relevant": value}))
        for field in ("score", "confidence", "type"):
            missing = response({"relevant": 2.8})
            del missing["answers"]["relevant"][field]
            bad.append(missing)
        for bad_response in bad:
            with self.subTest(response=bad_response):
                transport = Mock(return_value=bad_response)
                client = self.client(transport)
                with self.assertRaises(JevError):
                    self.decide(client)
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(client.events[-1]["status"], "failed")

    def test_invalid_score_request_is_rejected_before_transport(self):
        bad_questions = [
            {"type": "noul", "instructions": "Rate this", "criteria": CRITERIA},
            {"type": "score", "instructions": "Rate this"},
            {"type": "score", "instructions": "Rate this", "criteria": {"0": "None"}},
            {"type": "score", "instructions": "Rate this", "criteria": ["No", 2]},
            {"type": "score", "instructions": "", "criteria": CRITERIA},
        ]
        for question in bad_questions:
            with self.subTest(question=question):
                transport = Mock()
                with self.assertRaises(ValueError):
                    self.client(transport).decide(
                        {"question": "Current ratio?"}, {"relevant": question},
                        call_id="bad-request", stage="test")
                transport.assert_not_called()

    def test_transient_error_retries_once_and_records_both_attempts(self):
        transport = Mock(side_effect=[URLError("connection lost"), response({"relevant": 0.7})])
        client = self.client(transport)
        self.assertEqual(self.decide(client), {
            "relevant": {"type": "score", "score": 0.7, "confidence": 0.8}})
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
                self.assertEqual(question["type"], "score")
                self.assertEqual(question["instructions"], (
                    f"Rate how much candidates[{index}] supports solving the question using the ordered criteria. "
                    "Judge relevance and usability only; do not solve the question or verify arithmetic."
                ))
                self.assertEqual(question["criteria"], CRITERIA)
            by_id = {"calc-00001": 2.95, "calc-00002": 0.1, "err-00003": 2.8}
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
        self.assertEqual(metadata["primitive"], "score")
        self.assertEqual(metadata["ranking"], "score_desc_then_playbook_order")
        self.assertEqual(metadata["scores"], {"calc-00001": 2.95, "calc-00002": 0.1, "err-00003": 2.8})
        self.assertEqual(metadata["confidences"], {name: 0.8 for name in metadata["scores"]})
        self.assertEqual(client.events[-1], {"event": "read_selection", **metadata})

    def test_ties_preserve_original_order(self):
        def transport(request, timeout):
            questions = json.loads(request.data)["questions"]
            return response({name: 0.5 for name in questions})

        client = JevClient(enabled=True, api_key_env="TEST_JEV_KEY", transport=transport)
        selected, metadata = ReadSelector(client, 1, batch_size=1).select(
            PLAYBOOK, "A question", call_id="tie", stage="val")
        self.assertEqual(metadata["selected_ids"], ["calc-00001"])
        self.assertNotIn("[calc-00002]", selected)

    def test_confidence_never_changes_score_ranking_or_cross_batch_ties(self):
        values = {"calc-00001": (2.25, 0.01), "calc-00002": (2.25, 1.0),
                  "err-00003": (2.75, 0.0)}

        def transport(request, timeout):
            payload = json.loads(request.data)
            by_name = {name: values[candidate["id"]]
                       for name, candidate in zip(payload["questions"], payload["state"]["candidates"])}
            return response({name: pair[0] for name, pair in by_name.items()},
                            {name: pair[1] for name, pair in by_name.items()})

        client = JevClient(enabled=True, api_key_env="TEST_JEV_KEY", transport=transport)
        selected, metadata = ReadSelector(client, 2, batch_size=1).select(
            PLAYBOOK, "A question", call_id="confidence", stage="test")
        self.assertEqual(metadata["selected_ids"], ["err-00003", "calc-00001"])
        self.assertEqual(metadata["jev_calls"], 3)
        self.assertEqual(metadata["scores"]["calc-00001"], 2.25)
        self.assertEqual(metadata["confidences"]["calc-00002"], 1.0)
        self.assertNotIn("[calc-00002]", selected)
        self.assertLess(selected.index("[calc-00001]"), selected.index("[err-00003]"))

    def test_top_ten_keeps_raw_scores_stable_cross_batch_ties_and_original_structure(self):
        parts = ["Original preamble.\r\n"]
        for index in range(15):
            if index % 5 == 0:
                parts.append(f"## SECTION {index // 5}\r\n")
            parts.append(
                f"[calc-{index:05d}] helpful={index} harmful=0 :: Formula {index}.\r\n"
                f"  Original explanation {index}.\r\n")
        book = "".join(parts)
        values = [0.1, 2.001, 2.002, 2.003, 2.004, 2.005, 2.006, 2.007,
                  2.008, 2.009, 2.003, 2.011, 2.012, 0.1, 2.003]
        captured = []

        def transport(request, timeout):
            payload = json.loads(request.data)
            captured.append(payload)
            by_name = {name: int(candidate["id"].split("-")[-1])
                       for name, candidate in zip(payload["questions"], payload["state"]["candidates"])}
            payload = response({name: values[index] for name, index in by_name.items()},
                               {name: index / 14 for name, index in by_name.items()})
            for answer in payload["answers"].values():
                # An inconsistent optional total must not stop or alter ranking.
                answer["probabilities"] = {"0": 0, "1": 0.0, "2": 0.49, "3": 0.5}
            return payload

        client = JevClient(enabled=True, api_key_env="TEST_JEV_KEY", transport=transport)
        selected, metadata = ReadSelector(client, 10, batch_size=4).select(
            book, "Question", call_id="fractional-top-ten", stage="test")
        expected = (12, 11, 9, 8, 7, 6, 5, 4, 3, 10)
        self.assertEqual(metadata["selected_ids"], [f"calc-{index:05d}" for index in expected])
        self.assertEqual(metadata["selected_count"], 10)
        self.assertEqual(metadata["jev_calls"], 4)
        self.assertEqual([len(item["state"]["candidates"]) for item in captured], [4, 4, 4, 3])
        self.assertEqual(metadata["scores"], {f"calc-{index:05d}": value
                                              for index, value in enumerate(values)})
        self.assertGreater(metadata["confidences"]["calc-00014"], metadata["confidences"]["calc-00003"])
        self.assertNotIn("[calc-00014]", selected)
        selected_ids = {f"calc-{index:05d}" for index in expected}
        expected_book = "".join(part for part in parts
                                if not part.startswith("[calc-")
                                or part.split("]", 1)[0][1:] in selected_ids)
        self.assertEqual(selected, expected_book)

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
        ten_entries = "## FORMULAS\n" + "".join(
            f"[calc-{index:05d}] helpful=0 harmful=0 :: Formula {index}.\n"
            for index in range(10))
        for book in ("## OTHERS\n", PLAYBOOK, ten_entries):
            selected, metadata = ReadSelector(client, 10).select(
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
