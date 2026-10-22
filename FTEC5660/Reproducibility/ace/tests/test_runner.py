"""Whole-protocol offline checks, including real network/client guards."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import runner


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def read_summary(path):
    with (Path(path) / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def offline_guards():
    """Make any accidental socket, HTTP call, or real model client fail."""
    guards = []
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "bailian-unusable-test-sentinel",
            "DASHSCOPE_API_KEY": "unused-dashscope-test-sentinel",
            "SAMBANOVA_API_KEY": "unused-sambanova-test-sentinel",
            "TYPESAFE_API_KEY": "unusable-test-sentinel",
        }, clear=True))
        for target in (
            "socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
            "socket.getaddrinfo", "urllib.request.OpenerDirector.open",
            "http.client.HTTPConnection.connect", "http.client.HTTPSConnection.connect",
            "openai.OpenAI",
        ):
            guards.append(stack.enter_context(patch(
                target, side_effect=AssertionError(f"Offline check attempted {target}"))))
        stack.enter_context(patch("dotenv.load_dotenv", return_value=False))
        stack.enter_context(patch("tiktoken.get_encoding", side_effect=AssertionError("Unexpected tokenizer fetch")))
        yield guards
        for guard in guards:
            guard.assert_not_called()


class DefaultPlanTests(unittest.TestCase):
    def test_default_plan_needs_no_site_packages_credentials_or_network(self):
        # -S disables site packages, independently checking that the default
        # command does not rely on SDKs that happen to be installed for tests.
        child = """
import runpy
import socket
import sys
import urllib.request
def forbidden(*args, **kwargs):
    raise AssertionError('default plan attempted network access')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
urllib.request.OpenerDirector.open = forbidden
sys.argv = ['eval.finance.experiments', '--output', sys.argv[1]]
runpy.run_module('eval.finance.experiments', run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan"
            completed = subprocess.run(
                [sys.executable, "-S", "-c", child, str(output)], cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            printed = json.loads(completed.stdout)
            self.assertEqual(printed["mode"], "dry-run")
            self.assertEqual(printed["samples"], {"train": 100, "val": 50, "test": 200})
            self.assertEqual(printed["groups"], list("ABC"))
            self.assertEqual(printed["required_keys_for_live_run"], ["DEEPSEEK_API_KEY", "TYPESAFE_API_KEY"])
            plan = read_json(output / "plan.json")
            self.assertEqual(plan["schema_version"], 8)
            self.assertEqual(plan["protocol"], "formula-abc-compact-artifacts-bailian-v8")
            self.assertEqual(plan["config"]["jev"]["k"], 5)
            self.assertNotIn("k_candidates", plan["config"]["jev"])
            llm = plan["config"]["llm"]
            self.assertEqual(llm["api_provider"], "dashscope")
            self.assertEqual(llm["model"], "siliconflow/deepseek-v3.1-terminus")
            self.assertEqual(llm["api_key_env"], "DEEPSEEK_API_KEY")
            self.assertIs(llm["enable_thinking"], False)
            self.assertEqual({path.name for path in output.iterdir()}, {"plan.json"})
            self.assertEqual(plan["data"]["sources"]["test"]["selected_count"], 200)
            self.assertFalse(any(output.glob("trial_*")))


class MockProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "three-arms"
        captured = io.StringIO()
        from src.experiments.runtime import Runtime
        original_response = Runtime._mock_response

        def expanded_book_response(runtime, role, call_id):
            # Six learned entries make fixed top-5 exercise the scoring path.
            response = json.loads(original_response(runtime, role, call_id))
            if role == "curator":
                operation = response["operations"][0]
                response["operations"] = [
                    {**operation, "content": f"{operation['content']} Rule {index}."}
                    for index in range(3)
                ]
            return json.dumps(response)

        with offline_guards(), redirect_stdout(captured), redirect_stderr(captured), patch.object(
            Runtime, "_mock_response", expanded_book_response,
        ):
            status = runner.main(["--mock", "--output", str(cls.output)])
        if status != 0:
            cls.temporary.cleanup()
            raise AssertionError("Mock protocol did not complete:\n" + captured.getvalue())
        cls.plan = read_json(cls.output / "plan.json")
        cls.results = {group: read_json(cls.output / group / "01" / "result.json")
                       for group in "ABC"}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def folder(self, group):
        return self.output / group / "01"

    def test_all_three_groups_complete_and_remain_explicitly_mock(self):
        self.assertEqual(self.plan["mode"], "mock")
        self.assertFalse(self.plan["formal_metrics"])
        for group, result in self.results.items():
            with self.subTest(group=group):
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["mock"])
                self.assertFalse(result["formal_metrics"])
                self.assertEqual(result["test"]["total"], 2)
                self.assertEqual(result["test"]["api_failure_count"], 0)
                self.assertEqual(result["fingerprint"], self.plan["fingerprint"])
        summary = read_summary(self.output)
        self.assertEqual({row["group"] for row in summary}, set("ABC"))
        self.assertEqual({path.name for path in self.output.iterdir() if path.is_dir()}, set("ABC"))
        self.assertTrue(all((self.output / group / "01").is_dir() for group in "ABC"))
        self.assertFalse(any(self.output.glob("trial_*")))
        self.assertTrue(all(row["formal_metrics"] == "False" for row in summary))

    def test_final_predictions_are_flat_and_agree_with_saved_metrics(self):
        for group, result in self.results.items():
            with self.subTest(group=group):
                folder = self.folder(group)
                predictions = read_jsonl(folder / "predictions.jsonl")
                metrics = result["test"]
                self.assertNotIn("predictions", metrics)
                self.assertEqual(Path(result["predictions_path"]), (folder / "predictions.jsonl").resolve())
                self.assertEqual(len(predictions), metrics["total"])
                self.assertEqual(sum(row["is_correct"] for row in predictions), metrics["correct"])
                self.assertEqual([row["index"] for row in predictions], list(range(metrics["total"])))
                calls = read_jsonl(folder / "llm_calls.jsonl")
                call_keys = {(row["stage"], row["call_id"]) for row in calls}
                for prediction in predictions:
                    self.assertEqual(prediction["stage"], "test")
                    self.assertIn((prediction["stage"], prediction["call_id"]), call_keys)
                    self.assertTrue({"index", "sample_id", "prediction", "target", "is_correct", "api_success", "error"}.issubset(prediction))
                    self.assertTrue({"response", "final_answer", "success"}.isdisjoint(prediction))
                self.assertFalse((folder / "test_metrics.json").exists())
                self.assertFalse((folder / "evaluation_playbook.txt").exists())
                self.assertFalse((folder / "evaluation").exists())
                self.assertFalse((folder / "evaluation" / "test" / "predictions.jsonl").exists())
                self.assertFalse(list(folder.rglob("frozen_selection.json")))

    def test_only_b_trains_with_three_reflection_rounds(self):
        calls = read_jsonl(self.folder("B") / "llm_calls.jsonl")
        self.assertEqual(sum(row["role"] == "reflector" and row["stage"] == "training"
                             for row in calls), 6)
        self.assertNotIn("training", self.results["C"])
        self.assertFalse((self.folder("C") / "training").exists())

    def test_training_evidence_is_merged_with_validation_predictions_and_config(self):
        for group, expected_rounds in (("B", 3),):
            with self.subTest(group=group):
                result = self.results[group]
                training_dir = self.folder(group) / "training"
                self.assertEqual({path.name for path in training_dir.iterdir()},
                                 {"train_results.json", "best_playbook.txt"})
                training = read_json(training_dir / "train_results.json")
                self.assertEqual(training["status"], "complete")
                self.assertEqual(len(training["steps"]), 2)
                self.assertEqual(len(training["checkpoints"]), 1)
                for step in training["steps"]:
                    self.assertEqual(step["reflection_rounds_used"], expected_rounds)
                    self.assertIsNone(step["corrected_after_round"])
                    self.assertIn("is_correct", step["pre_train_result"])
                    self.assertIn("is_correct", step["post_train_result"])
                    self.assertIn("call_id", step["pre_train_result"])
                    self.assertIn("call_id", step["post_train_result"])
                self.assertEqual(training["best_checkpoint"]["playbook_sha256"], result["playbook_sha256"])
                self.assertEqual(result["training_config"]["effective_parameters"]["max_num_rounds"], expected_rounds)
                summary = result["training"]["training_results"]
                self.assertEqual(summary["steps_completed"], len(training["steps"]))
                self.assertEqual(summary["checkpoints_completed"], len(training["checkpoints"]))
                self.assertEqual(Path(summary["train_results_path"]), (training_dir / "train_results.json").resolve())
                self.assertNotIn("predictions", summary)
                calls = {(row["stage"], row["call_id"]) for row in read_jsonl(self.folder(group) / "llm_calls.jsonl")}
                checkpoint = training["checkpoints"][0]
                self.assertEqual(len(checkpoint["predictions"]), 2)
                for prediction in checkpoint["predictions"]:
                    self.assertIn((prediction["stage"], prediction["call_id"]), calls)
                    self.assertTrue({"response", "final_answer", "success"}.isdisjoint(prediction))

    def test_c_reuses_b_checkpoint_without_training_or_target_in_jev(self):
        b, c = self.results["B"], self.results["C"]
        self.assertEqual(c["playbook_sha256"], b["playbook_sha256"])
        self.assertEqual(c["b_checkpoint"]["sha256"], b["playbook_sha256"])
        self.assertEqual(c["b_checkpoint"]["group"], "B")
        self.assertFalse(c["b_checkpoint"]["training_cost_included_in_C_incremental_usage"])
        self.assertTrue(c["b_checkpoint"]["shared_learning_excludes_B_test"])
        shared = c["b_checkpoint"]["shared_learning_llm_usage"]
        self.assertEqual(shared["successful_calls"], b["llm"]["successful_calls"] - b["llm"]["stage_usage"]["test"]["successful_calls"])
        self.assertEqual(c["llm_usage_including_shared_learning"]["successful_calls"],
                         c["llm"]["successful_calls"] + shared["successful_calls"])
        self.assertFalse((self.folder("C") / "training").exists())
        self.assertNotIn("training", c)
        calls = read_jsonl(self.folder("C") / "llm_calls.jsonl")
        self.assertEqual(len(calls), self.plan["data"]["sources"]["test"]["selected_count"])
        self.assertTrue(all(row["role"] == "generator" for row in calls))
        self.assertTrue(all(row["stage"] == "test" for row in calls))
        self.assertEqual(set(c["llm"]["stage_usage"]), {"test"})
        self.assertFalse(list(self.folder("C").glob("validation*")))
        self.assertNotIn("validation_k_results", c)
        self.assertNotIn("chosen_k", c)
        events = read_jsonl(self.folder("C") / "jev_audit.jsonl")
        requests = [row for row in events if row["event"] == "jev_request"]
        self.assertGreater(len(requests), 0, "the mock book must exceed fixed k=5")
        self.assertTrue(all(row["stage"] == "test" for row in requests))
        for event in requests:
            self.assertEqual(set(event["state"]), {"question", "candidates"})
            self.assertNotIn("target", json.dumps(event["questions"]))
            self.assertEqual(event["model"], self.plan["config"]["jev"]["model"])
        self.assertEqual(c["selection_k"], 5)
        self.assertEqual(c["selection_method"], "fixed_top_k")
        selections = [row for row in events if row["event"] == "read_selection"]
        self.assertEqual(len(selections), len(calls))
        self.assertTrue(all(row["k"] == 5 and row["selected_count"] == 5 for row in selections))
        # The scored subset must be the actual context handed to DeepSeek.
        candidate_ids = {candidate["id"] for request in requests
                         for candidate in request["state"]["candidates"]}
        selection_by_call = {row["call_id"]: row for row in selections}
        for call in calls:
            selected_ids = set(selection_by_call[f"test:{call['call_id']}"]["selected_ids"])
            for bullet_id in candidate_ids:
                self.assertEqual(f"[{bullet_id}]" in call["prompt"], bullet_id in selected_ids)
        self.assertFalse((self.folder("C") / "frozen_selection.json").exists())
        self.assertEqual(Path(c["b_checkpoint"]["path"]),
                         (self.folder("B") / "training" / "best_playbook.txt").resolve())

    def test_c_selection_and_source_are_persisted_before_testing_and_survive_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
            source = output / "B" / "01"
            (source / "training").mkdir(parents=True)
            (source / "training" / "best_playbook.txt").write_bytes(
                (self.folder("B") / "training" / "best_playbook.txt").read_bytes())
            (source / "result.json").write_text(json.dumps(self.results["B"]), encoding="utf-8")
            plan = self.plan
            data = runner.prepare_data(plan["config"], ROOT)
            observed = []

            def fail_before_requests(*args, **kwargs):
                persisted = read_json(output / "C" / "01" / "result.json")
                self.assertEqual(persisted["status"], "running")
                self.assertEqual(persisted["selection_k"], 5)
                self.assertEqual(persisted["selection_method"], "fixed_top_k")
                self.assertEqual(persisted["b_checkpoint"]["sha256"], self.results["B"]["playbook_sha256"])
                self.assertEqual(persisted["playbook_sha256"], persisted["b_checkpoint"]["sha256"])
                self.assertEqual(Path(persisted["b_checkpoint"]["path"]).resolve(),
                                 (source / "training" / "best_playbook.txt").resolve())
                observed.append(persisted)
                raise RuntimeError("injected offline evaluation failure")

            with offline_guards(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch.object(
                runner, "checked_evaluation", side_effect=fail_before_requests,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected offline evaluation failure"):
                    runner.run_group("C", 1, plan, data, output, mock=True)
            self.assertEqual(len(observed), 1)
            failed = read_json(output / "C" / "01" / "result.json")
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["selection_k"], 5)
            self.assertEqual(failed["selection_method"], "fixed_top_k")
            self.assertEqual(failed["b_checkpoint"], observed[0]["b_checkpoint"])
            self.assertFalse((output / "C" / "01" / "frozen_selection.json").exists())

    def test_a_is_the_empty_playbook_baseline_and_every_group_omits_formula_definitions(self):
        a = self.results["A"]
        self.assertNotIn("training", a)
        self.assertFalse((self.folder("A") / "training").exists())
        baseline = read_jsonl(self.folder("A") / "llm_calls.jsonl")
        self.assertEqual(len(baseline), 2)
        self.assertTrue(all(row["role"] == "generator" and row["stage"] == "test" for row in baseline))
        from src import ACE
        empty_book = ACE._initialize_empty_playbook(None)
        self.assertIn("## FORMULAS & CALCULATIONS", empty_book)
        self.assertNotIn("helpful=", empty_book)
        self.assertTrue(all(empty_book in row["prompt"] for row in baseline))
        for group in "ABC":
            for call in read_jsonl(self.folder(group) / "llm_calls.jsonl"):
                self.assertEqual(call["max_tokens"], 4096)
                self.assertNotEqual(call["role"], "dataset_baseline")
                self.assertNotIn("Use formula ", call["prompt"])
                self.assertNotIn("Formula:", call["prompt"])
                self.assertNotIn("Explanation:", call["prompt"])
        events = read_jsonl(self.folder("C") / "jev_audit.jsonl")
        self.assertFalse(any(row["event"] == "write_gate" for row in events))

    def test_model_settings_and_token_budget_are_recorded_for_every_arm(self):
        model = self.plan["config"]["llm"]["model"]
        for group in "ABC":
            calls = read_jsonl(self.folder(group) / "llm_calls.jsonl")
            self.assertTrue(all(row["model"] == model and row["temperature"] == 0 for row in calls))
            self.assertTrue(all(row["api_provider"] == "dashscope" for row in calls))
            self.assertTrue(all(row["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1" for row in calls))
            self.assertTrue(all(row["enable_thinking"] is False for row in calls))
            self.assertTrue(all(row["response_model"] == model for row in calls))
            self.assertTrue(all(row["max_tokens"] == 4096 for row in calls))
            self.assertTrue(all(row["mock"] for row in calls))
            raw = (self.folder(group) / "llm_calls.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("unusable-test-sentinel", raw)
            self.assertNotIn("offline-placeholder", raw)

    def test_completed_output_cannot_be_overwritten(self):
        result_path = self.folder("B") / "result.json"
        before = result_path.read_bytes()
        captured = io.StringIO()
        with offline_guards(), redirect_stdout(captured), redirect_stderr(captured), patch.object(runner, "run_group") as run_group:
            with self.assertRaises(SystemExit) as failure:
                runner.main(["--mock", "--groups", "B", "--output", str(self.output)])
        self.assertEqual(failure.exception.code, 2)
        run_group.assert_not_called()
        self.assertEqual(result_path.read_bytes(), before)

    def test_standalone_c_missing_b_fails_before_any_trial_or_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "no-b"
            with offline_guards(), patch.object(runner, "run_group") as run_group:
                with self.assertRaisesRegex(ValueError, "completed B"):
                    runner.main(["--mock", "--groups", "C", "--output", str(output)])
            run_group.assert_not_called()
            self.assertFalse(output.exists())

    def test_b_checkpoint_hash_detects_a_changed_handbook(self):
        checkpoint = self.folder("B") / "training" / "best_playbook.txt"
        before = checkpoint.read_bytes()
        try:
            checkpoint.write_bytes(before + b"\nChanged after completion\n")
            with self.assertRaisesRegex(ValueError, "changed after completion"):
                runner.b_checkpoint(self.output, 1, self.plan)
        finally:
            checkpoint.write_bytes(before)
        self.assertEqual(hashlib.sha256(before).hexdigest(), self.results["B"]["playbook_sha256"])

    def test_c_rejects_a_checkpoint_claiming_another_group_or_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
            folder = output / "B" / "01"
            (folder / "training").mkdir(parents=True)
            (folder / "training" / "best_playbook.txt").write_bytes(
                (self.folder("B") / "training" / "best_playbook.txt").read_bytes())
            for changed in ({"group": "C"}, {"trial": 2}):
                with self.subTest(changed=changed):
                    (folder / "result.json").write_text(json.dumps({**self.results["B"], **changed}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "same trial"):
                        runner.b_checkpoint(output, 1, self.plan)

    def test_different_configuration_cannot_reuse_or_overwrite_completed_results(self):
        config = json.loads(json.dumps(self.plan["config"]))
        config["training"]["test_workers"] = 2
        result_path = self.folder("A") / "result.json"
        before = result_path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with offline_guards(), redirect_stderr(io.StringIO()), patch.object(runner, "run_group") as run_group:
                with self.assertRaises(SystemExit) as error:
                    runner.main(["--mock", "--groups", "A", "--config", str(path),
                                 "--output", str(self.output)])
            self.assertEqual(error.exception.code, 2)
            run_group.assert_not_called()
        self.assertEqual(result_path.read_bytes(), before)

    def test_aggregation_ignores_other_layouts_groups_and_out_of_range_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
            result = self.results["A"]
            for relative in ("A/01/result.json", "archive/01/result.json",
                             "trial_01/A/result.json", "A/01/nested/result.json",
                             "A/99/result.json", "D/01/result.json"):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result), encoding="utf-8")
            rows = runner.aggregate(output, self.plan)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group"], "A")
            self.assertEqual(rows[0]["completed_trials"], 1)
            saved = read_summary(output)
            self.assertEqual([(row["group"], row["completed_trials"]) for row in saved], [("A", "1")])
            self.assertFalse((output / "summary.json").exists())

    def test_removed_groups_and_original_baseline_cannot_be_selected(self):
        from src.experiments import runtime, jev
        self.assertEqual(tuple(runner.GROUPS), tuple("ABC"))
        self.assertFalse(hasattr(runtime, "RawGenerator"))
        self.assertFalse(hasattr(jev, "GatedCurator"))
        for group in ("D", "E", "F"):
            with self.subTest(group=group), offline_guards(), redirect_stderr(io.StringIO()):
                with patch.object(runner, "run_group") as run_group, self.assertRaises(SystemExit) as error:
                    runner.main(["--mock", "--groups", group])
                self.assertEqual(error.exception.code, 2)
                run_group.assert_not_called()


class StagedExecutionTests(unittest.TestCase):
    def test_three_trial_protocol_writes_only_the_39_planned_artifacts(self):
        config = runner.load_config(ROOT / "configs/formula/smoke.json")
        config["repeats"] = 3
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "three-repeats.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = Path(directory) / "results"
            with offline_guards(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["--mock", "--config", str(config_path), "--output", str(output)]), 0)
            expected = {"plan.json", "invocations.jsonl", "summary.csv"}
            for group in "ABC":
                for trial in ("01", "02", "03"):
                    prefix = f"{group}/{trial}"
                    expected.update(f"{prefix}/{name}" for name in
                                    ("llm_calls.jsonl", "predictions.jsonl", "result.json"))
                    if group == "B":
                        expected.update(f"{prefix}/training/{name}" for name in
                                        ("train_results.json", "best_playbook.txt"))
                    elif group == "C":
                        expected.add(f"{prefix}/jev_audit.jsonl")
            actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 39)
            self.assertEqual({row["group"]: row["completed_trials"] for row in read_summary(output)},
                             {group: "3" for group in "ABC"})
            for trial in ("01", "02", "03"):
                b = read_json(output / "B" / trial / "result.json")
                c = read_json(output / "C" / trial / "result.json")
                self.assertEqual(c["b_checkpoint"]["sha256"], b["playbook_sha256"])
                self.assertEqual(Path(c["b_checkpoint"]["path"]),
                                 (output / "B" / trial / "training" / "best_playbook.txt").resolve())

    def test_a_can_finish_three_trials_before_b_and_c_without_typesafe(self):
        config = runner.load_config(ROOT / "configs/formula/smoke.json")
        config["repeats"] = 3
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "three-repeats.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = Path(directory) / "results"
            common = ["--mock", "--config", str(config_path), "--output", str(output)]
            captured = io.StringIO()
            with offline_guards(), redirect_stdout(captured), redirect_stderr(captured):
                with patch.object(runner, "jev_client", side_effect=AssertionError("A must not construct JEV")):
                    self.assertEqual(runner.main(common + ["--groups", "A"]), 0)
                plan = read_json(output / "plan.json")
                self.assertEqual(plan["required_keys"], ["DEEPSEEK_API_KEY"])
                self.assertEqual({path.name for path in output.iterdir() if path.is_dir()}, {"A"})
                self.assertEqual({path.name for path in (output / "A").iterdir()}, {"01", "02", "03"})
                original_a = {}
                for trial in ("01", "02", "03"):
                    result = output / "A" / trial / "result.json"
                    original_a[trial] = result.read_bytes()
                    self.assertEqual(read_json(result)["status"], "complete")
                    self.assertEqual(len(read_jsonl(output / "A" / trial / "llm_calls.jsonl")), 2)
                    predictions = read_jsonl(output / "A" / trial / "predictions.jsonl")
                    self.assertEqual(len(predictions), 2)
                    self.assertEqual(sum(row["is_correct"] for row in predictions), read_json(result)["test"]["correct"])
                    self.assertNotIn("predictions", read_json(result)["test"])
                    self.assertFalse((output / "A" / trial / "evaluation" / "test" / "predictions.jsonl").exists())
                summary = read_summary(output)
                self.assertEqual([(row["group"], row["completed_trials"]) for row in summary], [("A", "3")])
                self.assertEqual(runner.main(common + ["--groups", "B", "--trial", "2"]), 0)
                self.assertEqual(runner.main(common + ["--groups", "C", "--trial", "2"]), 0)
                c = read_json(output / "C" / "02" / "result.json")
                b = read_json(output / "B" / "02" / "result.json")
                self.assertEqual(c["b_checkpoint"]["trial"], 2)
                self.assertEqual(Path(c["b_checkpoint"]["path"]), (output / "B" / "02" / "training" / "best_playbook.txt").resolve())
                self.assertEqual(c["playbook_sha256"], b["playbook_sha256"])
                self.assertFalse((output / "C" / "02" / "training").exists())
                self.assertFalse((output / "B" / "01").exists())
                self.assertFalse((output / "D").exists())
                with self.assertRaisesRegex(ValueError, "completed B"):
                    runner.main(common + ["--groups", "C", "--trial", "1"])
                self.assertFalse((output / "C" / "01").exists())
                self.assertEqual(original_a, {trial: (output / "A" / trial / "result.json").read_bytes()
                                              for trial in original_a})
                summary = read_summary(output)
                self.assertEqual({row["group"]: int(row["completed_trials"]) for row in summary},
                                 {"A": 3, "B": 1, "C": 1})

    def test_default_mock_and_real_plan_outputs_are_separate_and_group_first(self):
        original_prepare = runner.prepare_data
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            captured = io.StringIO()
            with offline_guards(), redirect_stdout(captured), redirect_stderr(captured), patch.object(
                runner, "PROJECT_ROOT", project,
            ), patch.object(runner, "prepare_data", side_effect=lambda config, _root: original_prepare(config, ROOT)):
                self.assertEqual(runner.main(["--mock", "--groups", "A"]), 0)
                self.assertEqual(runner.main(["--dry-run", "--groups", "A"]), 0)
            self.assertEqual({path.name for path in project.iterdir()}, {"results", "results-mock"})
            self.assertTrue((project / "results-mock" / "A" / "01" / "result.json").is_file())
            self.assertEqual(read_json(project / "results-mock" / "plan.json")["mode"], "mock")
            self.assertEqual(read_json(project / "results" / "plan.json")["mode"], "live")
            self.assertFalse((project / "results" / "A").exists())


if __name__ == "__main__":
    unittest.main()
