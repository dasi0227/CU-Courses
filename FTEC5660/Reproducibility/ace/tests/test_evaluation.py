"""Offline evaluation checks: fake generators only, with no credentials or network."""

import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_utils():
    spec = importlib.util.spec_from_file_location("offline_evaluation_utils", ROOT / "src/common/utils.py")
    module = importlib.util.module_from_spec(spec)
    # Do not inspect real .env values or fetch a tokenizer while loading tests.
    with patch("dotenv.load_dotenv", return_value=False), patch(
        "tiktoken.get_encoding", side_effect=AssertionError("Unexpected tokenizer load during import")
    ):
        spec.loader.exec_module(module)
    return module


utils = load_utils()


class FormulaProcessor:
    def answer_is_correct(self, prediction, target):
        try:
            return float(prediction.replace(",", "")) == float(target)
        except (TypeError, ValueError):
            return False

    def evaluate_accuracy(self, predictions, targets):
        return sum(self.answer_is_correct(a, b) for a, b in zip(predictions, targets)) / len(targets) if targets else 0.0


class FakeGenerator:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[kwargs["question"]]
        if isinstance(response, Exception):
            raise response
        return response, [], {}


def sample(index, target="15.25"):
    return {"id": f"test:{index}", "context": "", "question": str(index), "target": target}


class EvaluationTests(unittest.TestCase):
    def test_extract_plain_numbers_and_existing_formats(self):
        cases = {
            "15": "15",
            " 15.25 ": "15.25",
            "-.25": "-.25",
            "1.25e-3": "1.25e-3",
            "1,500.25": "1,500.25",
            "The final answer is: 15.25": "15.25",
            "The final answer is: 15.25.": "15.25",
            '{"final_answer": "15.25"}': "15.25",
            '{"final_answer": 15.25}': "15.25",
            "Finish[15.25]": "15.25",
            "The final answer is: \\boxed{15.25}": "15.25",
            '{"final_answer": "TagOne,TagTwo"}': "TagOne,TagTwo",
            "unparseable": "No final answer found",
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(utils.extract_answer(response), expected)

    def test_tokenizer_is_loaded_lazily_and_cached(self):
        module = load_utils()
        encoder = SimpleNamespace(encode=lambda value: value.split())
        with patch.object(module.tiktoken, "get_encoding", return_value=encoder) as get_encoding:
            self.assertEqual(module.count_tokens("one two"), 2)
            self.assertEqual(module.count_tokens("three"), 1)
            get_encoding.assert_called_once_with("cl100k_base")

    def test_mixed_failures_keep_full_denominator_and_predictions(self):
        generator = FakeGenerator({"0": "15.25", "1": RuntimeError("offline failure"), "2": "0"})
        results, errors = utils.evaluate_test_set(
            FormulaProcessor(), generator, "", [sample(i) for i in range(3)], max_workers=2
        )
        self.assertEqual(results["total"], 3)
        self.assertEqual(results["correct"], 1)
        self.assertAlmostEqual(results["accuracy"], 1 / 3)
        self.assertEqual(results["api_success_count"], 2)
        self.assertEqual(results["api_failure_count"], 1)
        self.assertAlmostEqual(results["success_rate"], 2 / 3)
        self.assertEqual(results["no_answer"], 1)
        self.assertEqual([record["sample_id"] for record in results["predictions"]], ["test:0", "test:1", "test:2"])
        failed = results["predictions"][1]
        self.assertFalse(failed["api_success"])
        self.assertFalse(failed["is_correct"])
        self.assertIn("offline failure", failed["error"])
        self.assertEqual(len(errors["errors"]), 2)

    def test_all_failures_and_empty_input_return_normally(self):
        generator = FakeGenerator({
            "0": RuntimeError("failure"),
            "1": "",
            "2": "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE",
        })
        results, _ = utils.evaluate_test_set(
            FormulaProcessor(), generator, "", [sample(0), sample(1), sample(2)], max_workers=2
        )
        self.assertEqual((results["accuracy"], results["correct"], results["total"], results["no_answer"]), (0, 0, 3, 3))
        self.assertEqual(results["api_success_count"], 0)
        self.assertEqual(results["api_failure_count"], 3)
        empty, _ = utils.evaluate_test_set(FormulaProcessor(), generator, "", [])
        self.assertEqual((empty["accuracy"], empty["total"], empty["predictions"]), (0, 0, []))

    def test_parallel_completion_is_returned_in_input_order(self):
        second_done = threading.Event()
        completion_order = []

        class OrderedGenerator:
            def generate(self, **kwargs):
                if kwargs["question"] == "0":
                    if not second_done.wait(timeout=2):
                        raise AssertionError("Second worker did not run")
                else:
                    completion_order.append(1)
                    second_done.set()
                if kwargs["question"] == "0":
                    completion_order.append(0)
                return "15.25", [], {}

        results, _ = utils.evaluate_test_set(
            FormulaProcessor(), OrderedGenerator(), "", [sample(0), sample(1)], max_workers=2
        )
        self.assertEqual(completion_order, [1, 0])
        self.assertEqual([row["index"] for row in results["predictions"]], [0, 1])

    def test_stages_save_separate_ordered_prediction_files(self):
        generator = FakeGenerator({"0": "15.25", "1": "15.25"})
        with tempfile.TemporaryDirectory() as directory:
            for stage in ("initial", "validation_e_1_s_10", "test"):
                utils.evaluate_test_set(
                    FormulaProcessor(), generator, "", [sample(0), sample(1)],
                    log_dir=directory, stage=stage, max_workers=2,
                )
                output = Path(directory) / stage / "predictions.jsonl"
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([row["sample_id"] for row in rows], ["test:0", "test:1"])
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {"initial", "validation_e_1_s_10", "test"})
            self.assertTrue(all(call["reflection"] == "(empty)" for call in generator.calls))
            self.assertTrue(all(Path(call["log_dir"]).parent == Path(directory) for call in generator.calls))

    def test_custom_predictions_path_keeps_errors_without_a_stage_duplicate(self):
        generator = FakeGenerator({"0": "15.25", "1": RuntimeError("offline failure"), "2": "0"})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "A" / "01"
            destination = folder / "predictions.jsonl"
            metrics, _ = utils.evaluate_test_set(
                FormulaProcessor(), generator, "", [sample(i) for i in range(3)],
                log_dir=folder / "evaluation", stage="test", max_workers=2,
                predictions_path=destination,
            )
            rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, metrics["predictions"])
            self.assertEqual([row["sample_id"] for row in rows], ["test:0", "test:1", "test:2"])
            self.assertEqual((len(rows), metrics["total"], metrics["api_failure_count"]), (3, 3, 1))
            self.assertFalse(rows[1]["api_success"])
            self.assertIn("offline failure", rows[1]["error"])
            stage_logs = folder / "evaluation" / "test"
            self.assertTrue(stage_logs.is_dir())
            self.assertTrue(all(Path(call["log_dir"]) == stage_logs for call in generator.calls))
            self.assertFalse((stage_logs / "predictions.jsonl").exists())

    def test_custom_predictions_path_does_not_require_a_log_directory(self):
        generator = FakeGenerator({"0": "15.25"})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "A" / "01" / "predictions.jsonl"
            metrics, _ = utils.evaluate_test_set(
                FormulaProcessor(), generator, "", [sample(0)],
                predictions_path=destination,
            )
            rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, metrics["predictions"])
            self.assertIsNone(generator.calls[0]["log_dir"])

    def test_compact_predictions_link_success_and_failure_to_raw_calls(self):
        generator = FakeGenerator({"0": "15.25", "1": RuntimeError("offline failure"), "2": "0"})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "A" / "01" / "predictions.jsonl"
            metrics, errors = utils.evaluate_test_set(
                FormulaProcessor(), generator, "", [sample(i) for i in range(3)],
                stage="test", predictions_path=destination, compact_predictions=True,
            )
            rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, metrics["predictions"])
            self.assertEqual((metrics["total"], metrics["correct"], metrics["api_failure_count"]), (3, 1, 1))
            expected_fields = {"index", "sample_id", "call_id", "stage", "prediction", "target",
                               "is_correct", "api_success", "error"}
            self.assertTrue(all(set(row) == expected_fields for row in rows))
            self.assertEqual([row["call_id"] for row in rows], ["test_eval_0", "test_eval_1", "test_eval_2"])
            self.assertTrue(all(row["stage"] == "test" for row in rows))
            self.assertEqual([row["call_id"] for row in rows], [call["call_id"] for call in generator.calls])
            self.assertFalse(rows[1]["api_success"])
            self.assertIn("offline failure", rows[1]["error"])
            self.assertEqual(len(errors["errors"]), 2)
            self.assertEqual({str(path.relative_to(Path(directory))) for path in Path(directory).rglob("*") if path.is_file()},
                             {"A/01/predictions.jsonl"})
            self.assertTrue(all(call["log_dir"] is None for call in generator.calls))

    def test_compact_in_memory_validation_keeps_one_prediction_representation(self):
        generator = FakeGenerator({"0": "15.25"})
        metrics, _ = utils.evaluate_test_set(
            FormulaProcessor(), generator, "", [sample(0)],
            stage="validation_e_1_s_20", compact_predictions=True,
        )
        prediction = metrics["predictions"][0]
        self.assertEqual(prediction["stage"], "validation_e_1_s_20")
        self.assertEqual(prediction["call_id"], "test_eval_0")
        self.assertNotIn("response", prediction)
        self.assertNotIn("final_answer", prediction)
        self.assertNotIn("success", prediction)

    def test_legacy_evaluation_keeps_response_fields_without_compact_opt_in(self):
        generator = FakeGenerator({"0": "15.25"})
        metrics, _ = utils.evaluate_test_set(FormulaProcessor(), generator, "", [sample(0)])
        prediction = metrics["predictions"][0]
        self.assertEqual(prediction["response"], "15.25")
        self.assertEqual(prediction["final_answer"], "15.25")
        self.assertTrue(prediction["success"])


if __name__ == "__main__":
    unittest.main()
