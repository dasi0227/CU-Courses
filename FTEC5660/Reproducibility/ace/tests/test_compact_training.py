"""Offline persistence checks for the opt-in compact ACE training record."""

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.ace import ACE


class Processor:
    @staticmethod
    def answer_is_correct(answer, target):
        return str(answer) == str(target)

    @classmethod
    def evaluate_accuracy(cls, answers, targets):
        return sum(cls.answer_is_correct(answer, target)
                   for answer, target in zip(answers, targets)) / len(targets)


def bare_engine():
    engine = object.__new__(ACE)
    engine.playbook = engine._initialize_empty_playbook()
    engine.best_playbook = engine.playbook
    engine.next_global_id = 1
    engine.max_tokens = 4096
    engine.use_bulletpoint_analyzer = False
    engine.bulletpoint_analyzer = None
    engine.generator = SimpleNamespace(model="offline", generate=Mock())
    engine.reflector = SimpleNamespace(model="offline", reflect=Mock(return_value=("fix", [], {})))
    engine.curator = SimpleNamespace(model="offline", curate=Mock())
    engine.curator.curate.side_effect = lambda **kwargs: (kwargs["current_playbook"], 1, [], {})
    return engine


class CompactTrainingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        for target in ("socket.create_connection", "socket.socket.connect", "openai.OpenAI"):
            guard = self.stack.enter_context(patch(target, side_effect=AssertionError("Network forbidden")))
            self.addCleanup(guard.assert_not_called)
        self.stack.enter_context(patch("src.ace.count_tokens", side_effect=len))

    def test_reflection_counts_only_error_correction_rounds(self):
        for answers, rounds, corrected in ((["1", "1"], 0, None),
                                           (["0", "0", "1", "1"], 2, 2),
                                           (["0", "0", "0", "0", "0"], 3, None)):
            with self.subTest(answers=answers):
                engine = bare_engine()
                engine.generator.generate.side_effect = [
                    (json.dumps({"final_answer": answer}), [], {}) for answer in answers
                ]
                params = engine._extract_config_params({"compact_artifacts": True, "max_num_rounds": 3})
                with patch("src.ace.log_bullet_usage") as usage:
                    _, _, tracking = engine._train_single_sample(
                        {"question": "offline", "context": "", "target": "1"}, Processor(),
                        "train_e_1_s_1", 1, 1, None, None, params, 1,
                    )
                usage.assert_not_called()
                self.assertEqual(tracking["reflection_rounds_used"], rounds)
                self.assertEqual(tracking["corrected_after_round"], corrected)
                self.assertEqual(tracking["pre_train_result"]["call_id"], "train_e_1_s_1_gen_initial")
                self.assertEqual(tracking["post_train_result"]["call_id"], "train_e_1_s_1_post_curate")
                for name in ("pre_train_result", "post_train_result"):
                    self.assertIn("final_answer", tracking[name])
                    self.assertIn("is_correct", tracking[name])
                    self.assertIn("playbook_num_tokens", tracking[name])

    def test_compact_record_has_one_validation_copy_and_only_best_book(self):
        engine = bare_engine()
        params = {"compact_artifacts": True, "eval_steps": 1, "save_steps": 1}
        targets = [{"sample_id": "train:1", "target": "1"},
                   {"sample_id": "train:2", "target": "1"}]
        observations = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def train(**kwargs):
                # The previous completed sample must already be on disk.
                saved = json.loads((root / "train_results.json").read_text())
                observations.append(len(saved["steps"]))
                engine.playbook = f"book-{kwargs['step']}"
                return "0", "1", {
                    "reflection_rounds_used": kwargs["step"],
                    "corrected_after_round": kwargs["step"],
                    "pre_train_result": {"final_answer": "0", "is_correct": False},
                    "post_train_result": {"final_answer": "1", "is_correct": True},
                }

            def validate(*args, **kwargs):
                self.assertIsNone(args[5])
                self.assertTrue(kwargs["compact_predictions"])
                if kwargs["stage"].endswith("s_2"):
                    self.assertEqual((root / "best_playbook.txt").read_text(), "book-1")
                return {"accuracy": 0.0, "total": 1, "correct": 0, "predictions": [{
                    "index": 0, "sample_id": "val:1", "call_id": "test_eval_0",
                    "stage": kwargs["stage"], "prediction": "0", "target": "1",
                    "is_correct": False, "api_success": True, "error": None,
                    "response": "raw answer stays in llm_calls", "final_answer": "0", "success": False,
                }]}, {"errors": []}

            with patch.object(engine, "_train_single_sample", side_effect=train), patch(
                "src.ace.evaluate_test_set", side_effect=validate,
            ):
                summary = engine._offline_train(targets, [{}], Processor(), params,
                                                directory, None, None, None)
            saved = json.loads((root / "train_results.json").read_text())
            self.assertEqual({path.name for path in root.iterdir()},
                             {"train_results.json", "best_playbook.txt"})
            self.assertEqual(observations, [0, 1])
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(len(saved["steps"]), 2)
            self.assertEqual(len(saved["checkpoints"]), 2)
            self.assertEqual(saved["best_checkpoint"]["step"], 1)
            self.assertEqual(saved["best_checkpoint"]["playbook_sha256"],
                             hashlib.sha256(b"book-1").hexdigest())
            self.assertEqual((root / "best_playbook.txt").read_text(), "book-1")
            for checkpoint in saved["checkpoints"]:
                self.assertNotIn("predictions", checkpoint["validation"])
                record = checkpoint["predictions"][0]
                self.assertIsNone(record["error"])
                self.assertFalse(set(record) & {"response", "final_answer", "success"})
            self.assertEqual(set(summary), {"best_validation_accuracy", "best_checkpoint",
                                           "steps_completed", "checkpoints_completed", "train_results_path"})
            self.assertEqual(summary["best_validation_accuracy"], 0.0)

    def test_failed_validation_is_saved_without_selecting_it_as_best(self):
        engine = bare_engine()
        tracking = {"reflection_rounds_used": 1, "corrected_after_round": None,
                    "pre_train_result": {"final_answer": "0", "is_correct": False},
                    "post_train_result": {"final_answer": "0", "is_correct": False}}
        validation = {"accuracy": 0.0, "total": 1, "correct": 0, "api_failure_count": 1,
                      "predictions": [{"index": 0, "sample_id": "val:1",
                                       "call_id": "test_eval_0", "stage": "validation_e_1_s_1",
                                       "prediction": "NO_ANSWER", "target": "1",
                                       "is_correct": False, "api_success": False,
                                       "error": "offline API failure"}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            engine, "_train_single_sample", return_value=("0", "0", tracking),
        ), patch("src.ace.evaluate_test_set", return_value=(validation, {"errors": []})):
            with self.assertRaisesRegex(RuntimeError, "Validation had API failures"):
                engine._offline_train([{"sample_id": "train:1", "target": "1"}], [{}],
                                      Processor(), {"compact_artifacts": True, "eval_steps": 1},
                                      directory, None, None, None)
            saved = json.loads((Path(directory) / "train_results.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(len(saved["steps"]), 1)
            self.assertEqual(len(saved["checkpoints"]), 1)
            self.assertEqual(saved["checkpoints"][0]["validation"]["api_failure_count"], 1)
            self.assertEqual(saved["checkpoints"][0]["predictions"][0]["error"], "offline API failure")
            self.assertEqual(saved["best_accuracy"], -1.0)
            self.assertIsNone(saved["best_checkpoint"])
            self.assertFalse((Path(directory) / "best_playbook.txt").exists())

    def test_failed_run_retains_completed_steps_and_failure_status(self):
        engine = bare_engine()
        first = ("0", "0", {"reflection_rounds_used": 1, "corrected_after_round": None,
                              "pre_train_result": {"final_answer": "0", "is_correct": False},
                              "post_train_result": {"final_answer": "0", "is_correct": False}})
        with tempfile.TemporaryDirectory() as directory, patch.object(
            engine, "_train_single_sample", side_effect=[first, RuntimeError("offline failure")],
        ):
            with self.assertRaisesRegex(RuntimeError, "offline failure"):
                engine._offline_train([{"sample_id": "train:1"}, {"sample_id": "train:2"}], [],
                                      Processor(), {"compact_artifacts": True, "eval_steps": 20},
                                      directory, None, None, None)
            saved = json.loads((Path(directory) / "train_results.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(len(saved["steps"]), 1)
            self.assertEqual(saved["steps"][0]["sample_id"], "train:1")
            self.assertEqual(saved["error"], {"type": "RuntimeError", "message": "offline failure"})

    def test_run_only_skips_legacy_summary_files_when_opted_in(self):
        for compact in (False, True):
            with self.subTest(compact=compact), tempfile.TemporaryDirectory() as directory:
                engine = bare_engine()
                with patch.object(engine, "_setup_paths", return_value=(directory, None, None, directory)), patch.object(
                    engine, "_offline_train", return_value={"best_validation_accuracy": 0.0},
                ):
                    engine.run("offline", [], [], data_processor=Processor(),
                               config={"compact_artifacts": compact})
                files = {path.name for path in Path(directory).iterdir()}
                self.assertEqual(files, set() if compact else {"run_config.json", "final_results.json"})


if __name__ == "__main__":
    unittest.main()
