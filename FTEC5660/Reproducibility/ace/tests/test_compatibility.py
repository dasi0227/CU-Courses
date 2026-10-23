"""Offline checks that ABC history compatibility is explicit and content bound."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import compatibility, runner


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class LegacyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "results"
        self.output.mkdir()
        config = runner.load_config(ROOT / "configs/formula/smoke.json")
        config["repeats"] = 3
        data = runner.prepare_data(config, ROOT)
        self.plan = runner.build_plan(config, data, list("ABC"), mock=True)
        self.plan.pop("compatibility", None)
        old_config = deepcopy(config)
        old_config["jev"]["k"] = 5
        old_config["groups"] = {
            "A": deepcopy(config["groups"]["A"]),
            "B": deepcopy(config["groups"]["B"]),
            "C": {"enabled": True, "kind": "ace", "max_num_rounds": 1},
            "D": deepcopy(config["groups"]["C"]),
        }
        old_source = deepcopy(self.plan["source_sha256"])
        # This is a synthetic v7 snapshot, never a rewritten production result.
        old_source["src/experiments/runner.py"] = hashlib.sha256(b"synthetic v7 runner").hexdigest()
        old_source.pop("src/experiments/compatibility.py", None)
        old_source["src/experiments/jev.py"] = hashlib.sha256(b"synthetic v7 Noul selector").hexdigest()
        self.old_plan = {
            "schema_version": 7,
            "protocol": "formula-abcd-compact-artifacts-bailian-v7",
            "config": old_config,
            "data": deepcopy(self.plan["data"]),
            "source_sha256": old_source,
            "mode": "mock",
            "formal_metrics": False,
            "groups": list("ABCD"),
        }
        self.old_plan["fingerprint"] = runner.digest({
            "config": old_config, "data": self.old_plan["data"],
            "source": old_source, "mock": True,
        })
        runner.write_json(self.output / "plan.json", self.old_plan)
        (self.output / "summary.csv").write_text("historical,summary\n", encoding="utf-8")
        (self.output / "invocations.jsonl").write_text('{"historical": true}\n', encoding="utf-8")
        preserved = {}
        for group in "AB":
            prefix = self.output / group / "01"
            prefix.mkdir(parents=True)
            result = {
                "group": group, "trial": 1, "status": "complete", "mock": True,
                "formal_metrics": False, "fingerprint": self.old_plan["fingerprint"],
                "test": {"accuracy": 0.0, "correct": 0, "total": 2},
                "wall_seconds": 1.0,
                "llm": {"successful_calls": 0, "attempts": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "stage_usage": {
                            "training": {"successful_calls": 0, "attempts": 0,
                                         "prompt_tokens": 0, "completion_tokens": 0}}},
                "llm_usage_including_shared_learning": {"successful_calls": 0,
                    "prompt_tokens": 0, "completion_tokens": 0},
                "jev": {"successful_calls": 0},
            }
            (prefix / "llm_calls.jsonl").write_text("{\"mock\": true}\n", encoding="utf-8")
            (prefix / "predictions.jsonl").write_text("{\"is_correct\": false}\n", encoding="utf-8")
            names = ["llm_calls.jsonl", "predictions.jsonl", "result.json"]
            if group == "B":
                training = prefix / "training"
                training.mkdir()
                (training / "best_playbook.txt").write_text("## OTHERS\n", encoding="utf-8")
                result["playbook_sha256"] = sha256(training / "best_playbook.txt")
                runner.write_json(training / "train_results.json", {
                    "status": "complete", "best_checkpoint": {
                        "playbook_sha256": result["playbook_sha256"],
                    },
                })
                names.extend(["training/train_results.json", "training/best_playbook.txt"])
            runner.write_json(prefix / "result.json", result)
            preserved[f"{group}/01"] = {
                "group": group, "trial": 1,
                "origin_id": "v7", "original_status": result["status"],
                "original_fingerprint": result["fingerprint"],
                "files_sha256": {name: sha256(prefix / name) for name in names},
            }
        changes = {
            name: {"old_sha256": old_source.get(name), "new_sha256": self.plan["source_sha256"].get(name)}
            for name in set(old_source) | set(self.plan["source_sha256"])
            if old_source.get(name) != self.plan["source_sha256"].get(name)
        }
        self.v9_plan = deepcopy(self.old_plan)
        self.v9_plan["schema_version"] = 9
        self.v9_plan["protocol"] = "formula-abc-score-compact-artifacts-bailian-v9"
        self.v9_plan["config"]["groups"] = deepcopy(config["groups"])
        self.v9_plan["groups"] = list("ABC")
        self.v9_plan["source_sha256"] = deepcopy(self.plan["source_sha256"])
        self.v9_plan["source_sha256"]["src/experiments/runner.py"] = hashlib.sha256(b"synthetic v9 runner").hexdigest()
        self.v9_plan["source_sha256"]["src/experiments/jev.py"] = hashlib.sha256(b"synthetic v9 top5 selector").hexdigest()
        self.v9_plan["fingerprint"] = runner.digest({
            "config": self.v9_plan["config"], "data": self.v9_plan["data"],
            "source": self.v9_plan["source_sha256"], "mock": True,
        })
        runner.write_json(self.output / "plan.v9.json", self.v9_plan)
        for group, trial, status in (("A", 2, "failed"), ("A", 3, "failed"), ("B", 2, "complete")):
            original = self.output / group / "01"
            prefix = self.output / group / f"{trial:02d}"
            for path in original.rglob("*"):
                if path.is_file():
                    destination = prefix / path.relative_to(original)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(path.read_bytes())
            result = read_json(prefix / "result.json")
            result.update(trial=trial, status=status, fingerprint=self.v9_plan["fingerprint"])
            if group == "B":
                result["recovery"] = {"transport_timeout_override": 180,
                                      "original_configured_timeout": 60,
                                      "offline_replayed_calls": 310}
            runner.write_json(prefix / "result.json", result)
            names = preserved[f"{group}/01"]["files_sha256"]
            preserved[f"{group}/{trial:02d}"] = {
                "origin_id": "v9", "group": group, "trial": trial,
                "original_status": status, "original_fingerprint": result["fingerprint"],
                "files_sha256": {name: sha256(prefix / name) for name in names},
            }
        self.manifest = {
            "schema_version": 2,
            "migration_id": "offline-multisource-score-top10-v10-fixture",
            "origin_plans": {
                "v7": {"path": "plan.json", "schema_version": 7,
                       "protocol": self.old_plan["protocol"],
                       "fingerprint": self.old_plan["fingerprint"],
                       "plan_sha256": sha256(self.output / "plan.json")},
                "v9": {"path": "plan.v9.json", "schema_version": 9,
                       "protocol": self.v9_plan["protocol"],
                       "fingerprint": self.v9_plan["fingerprint"],
                       "plan_sha256": sha256(self.output / "plan.v9.json")},
            },
            "target": {key: deepcopy(self.plan[key]) for key in (
                "schema_version", "protocol", "fingerprint", "config", "data",
                "source_sha256", "mode", "formal_metrics",
            )},
            "source_changes": {"v7": changes, "v9": {
                name: {"old_sha256": self.v9_plan["source_sha256"].get(name),
                       "new_sha256": self.plan["source_sha256"].get(name)}
                for name in set(self.v9_plan["source_sha256"]) | set(self.plan["source_sha256"])
                if self.v9_plan["source_sha256"].get(name) != self.plan["source_sha256"].get(name)
            }},
            "preserved_results": preserved,
        }
        self.manifest_path = Path(self.temporary.name) / "compatibility.json"
        runner.write_json(self.manifest_path, self.manifest)
        self.original_bytes = {
            str(path.relative_to(self.output)): path.read_bytes()
            for path in self.output.rglob("*") if path.is_file()
        }

    def load(self):
        return compatibility.load_compatibility(self.output, self.plan, manifest_path=self.manifest_path)

    def origin(self, group="B", trial=1, context=None):
        result = read_json(self.output / group / f"{trial:02d}" / "result.json")
        return compatibility.validate_result_origin(
            self.output, group, trial, self.plan, result, compatibility=context,
        )

    def test_pinned_history_is_recognized_without_rewriting_old_fingerprints_or_files(self):
        context = self.load()
        self.assertIsNotNone(context)
        for group in "AB":
            origin = self.origin(group, context=context)
            self.assertEqual(origin["kind"], "legacy_compatible")
            self.assertEqual(origin["fingerprint"], self.old_plan["fingerprint"])
        self.assertNotEqual(self.old_plan["fingerprint"], self.plan["fingerprint"])
        self.assertEqual(self.original_bytes, {
            str(path.relative_to(self.output)): path.read_bytes()
            for path in self.output.rglob("*") if path.is_file()
        })

    def test_legacy_output_uses_versioned_plan_invocation_and_summary_paths(self):
        context = self.load()
        self.assertEqual(compatibility.storage_names(self.output, self.plan, context),
                         ("plan.v10.json", "invocations.v10.jsonl", "summary.v10.csv"))
        fresh = Path(self.temporary.name) / "fresh"
        self.assertEqual(compatibility.storage_names(fresh, self.plan, None),
                         ("plan.json", "invocations.jsonl", "summary.csv"))

    def test_v8_history_is_not_implicitly_accepted_by_the_v10_migration(self):
        altered = deepcopy(self.old_plan)
        altered["schema_version"] = 8
        altered["protocol"] = "formula-abc-compact-artifacts-bailian-v8"
        runner.write_json(self.output / "plan.json", altered)
        with self.assertRaisesRegex(ValueError, "unrecognized historical plan"):
            self.load()

    def test_target_protocol_cannot_be_rebound_to_a_different_experiment(self):
        for schema, protocol in ((8, "formula-abc-compact-artifacts-bailian-v8"),
                                 (9, "formula-abc-noul-compact-artifacts-bailian-v9")):
            with self.subTest(schema=schema, protocol=protocol):
                target = deepcopy(self.plan)
                target["schema_version"] = schema
                target["protocol"] = protocol
                manifest = deepcopy(self.manifest)
                manifest["target"]["schema_version"] = schema
                manifest["target"]["protocol"] = protocol
                runner.write_json(self.manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "ABC Score top-10 v10 protocol"):
                    compatibility.load_compatibility(self.output, target, self.manifest_path)

    def test_v8_manifest_cannot_implicitly_approve_the_score_target(self):
        altered = deepcopy(self.manifest)
        altered["target"]["schema_version"] = 8
        altered["target"]["protocol"] = "formula-abc-compact-artifacts-bailian-v8"
        runner.write_json(self.manifest_path, altered)
        with self.assertRaisesRegex(ValueError, "changed after review"):
            self.load()

    def test_score_selector_source_is_bound_to_the_exact_reviewed_hash(self):
        self.assertIn("src/experiments/jev.py", self.manifest["source_changes"]["v7"])
        self.assertIsNotNone(self.load())
        altered = deepcopy(self.manifest)
        altered["source_changes"]["v7"]["src/experiments/jev.py"]["new_sha256"] = "0" * 64
        runner.write_json(self.manifest_path, altered)
        with self.assertRaisesRegex(ValueError, "source change review"):
            self.load()

    def test_each_historical_file_is_content_bound_including_predictions_and_best_book(self):
        for relative, before in self.original_bytes.items():
            if relative.count("/") < 2:
                continue
            with self.subTest(relative=relative):
                path = self.output / relative
                path.write_bytes(before + b"\nchanged\n")
                with self.assertRaises(ValueError):
                    self.load()
                path.write_bytes(before)

    def test_old_plan_content_and_fingerprint_must_both_verify(self):
        old_path = self.output / "plan.json"
        for mutation in (
            lambda p: p.update(fingerprint="0" * 64),
            lambda p: p["config"]["llm"].update(timeout=181),
            lambda p: p.update(schema_version=8),
        ):
            with self.subTest(mutation=mutation):
                altered = deepcopy(self.old_plan)
                mutation(altered)
                runner.write_json(old_path, altered)
                with self.assertRaises(ValueError):
                    self.load()
                old_path.write_bytes(self.original_bytes["plan.json"])

    def test_rebinding_the_old_plan_file_hash_does_not_bypass_its_own_fingerprint(self):
        altered = deepcopy(self.old_plan)
        altered["config"]["llm"]["timeout"] = 181
        runner.write_json(self.output / "plan.json", altered)
        manifest = deepcopy(self.manifest)
        manifest["origin_plans"]["v7"]["plan_sha256"] = sha256(self.output / "plan.json")
        runner.write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "self-consistent"):
            self.load()

    def test_arbitrary_target_config_data_sources_or_modes_cannot_reuse_history(self):
        mutations = (
            lambda p: p["config"]["llm"].update(timeout=181),
            lambda p: p["data"]["sources"]["test"].update(sha256="0" * 64),
            lambda p: p["source_sha256"].update({"src/experiments/runner.py": "0" * 64}),
            lambda p: p.update(mode="live", formal_metrics=True),
            lambda p: p.update(fingerprint="0" * 64),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                before = self.plan
                self.plan = deepcopy(before)
                mutation(self.plan)
                with self.assertRaises(ValueError):
                    self.load()
                self.plan = before

    def test_unlisted_trials_and_removed_c_or_d_are_not_legacy_compatible(self):
        context = self.load()
        original = read_json(self.output / "B" / "01" / "result.json")
        for group, trial in (("B", 3), ("C", 1), ("D", 1)):
            with self.subTest(group=group, trial=trial):
                result = {**original, "group": group, "trial": trial}
                runner.write_json(self.output / group / f"{trial:02d}" / "result.json", result)
                with self.assertRaises(ValueError):
                    compatibility.validate_result_origin(
                        self.output, group, trial, self.plan, result, compatibility=context,
                    )

    def test_runner_reuses_the_reviewed_b_for_new_c_without_touching_any_history(self):
        from tests.test_runner import offline_guards
        config_path = Path(self.temporary.name) / "smoke.json"
        runner.write_json(config_path, self.plan["config"])
        with offline_guards(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch.object(
            compatibility, "MANIFEST_PATH", self.manifest_path,
        ):
            book, lineage = runner.b_checkpoint(self.output, 1, self.plan)
            self.assertEqual(book, "## OTHERS\n")
            self.assertEqual(lineage["origin"]["kind"], "legacy_compatible")
            self.assertEqual(lineage["original_fingerprint"], self.old_plan["fingerprint"])
            self.assertFalse(lineage["training_cost_included_in_C_incremental_usage"])
            with self.assertRaisesRegex(ValueError, "complete plan"):
                runner.b_checkpoint(self.output, 1, self.plan["fingerprint"])
            self.assertEqual(runner.main([
                "--mock", "--groups", "C", "--trial", "1",
                "--config", str(config_path), "--output", str(self.output),
            ]), 0)
        result = read_json(self.output / "C" / "01" / "result.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["fingerprint"], self.plan["fingerprint"])
        self.assertEqual(result["b_checkpoint"]["origin"]["kind"], "legacy_compatible")
        self.assertEqual(result["b_checkpoint"]["original_fingerprint"], self.old_plan["fingerprint"])
        self.assertEqual(result["selection_k"], 10)
        self.assertNotIn("training", result)
        self.assertFalse((self.output / "C" / "01" / "training").exists())
        self.assertTrue((self.output / "plan.v10.json").is_file())
        self.assertTrue((self.output / "invocations.v10.jsonl").is_file())
        self.assertTrue((self.output / "summary.v10.csv").is_file())
        self.assertEqual(self.original_bytes, {
            name: (self.output / name).read_bytes() for name in self.original_bytes
        })

    def test_a_consistently_rebound_target_still_cannot_change_the_approved_experiment(self):
        self.plan["config"]["llm"]["timeout"] = 181
        self.plan["fingerprint"] = runner.digest({
            "config": self.plan["config"], "data": self.plan["data"],
            "source": self.plan["source_sha256"], "mock": True,
        })
        manifest = deepcopy(self.manifest)
        manifest["target"] = {key: deepcopy(self.plan[key]) for key in manifest["target"]}
        runner.write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "only removing old C"):
            self.load()

    def test_prompt_source_changes_cannot_be_whitelisted_by_rebinding_the_manifest(self):
        self.plan["source_sha256"]["src/core/generator.py"] = "0" * 64
        self.plan["fingerprint"] = runner.digest({
            "config": self.plan["config"], "data": self.plan["data"],
            "source": self.plan["source_sha256"], "mock": True,
        })
        manifest = deepcopy(self.manifest)
        manifest["target"] = {key: deepcopy(self.plan[key]) for key in manifest["target"]}
        manifest["source_changes"]["v7"]["src/core/generator.py"] = {
            "old_sha256": self.old_plan["source_sha256"]["src/core/generator.py"],
            "new_sha256": "0" * 64,
        }
        runner.write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "unreviewed source"):
            self.load()

    def test_manifest_must_describe_exactly_the_approved_config_and_source_transition(self):
        mutations = (
            lambda m: m["target"]["config"]["groups"]["B"].update(max_num_rounds=1),
            lambda m: m["source_changes"].clear(),
            lambda m: m["origin_plans"]["v7"].update(fingerprint="0" * 64),
            lambda m: m["preserved_results"]["B/01"].update(original_fingerprint="0" * 64),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                altered = deepcopy(self.manifest)
                mutation(altered)
                runner.write_json(self.manifest_path, altered)
                with self.assertRaises(ValueError):
                    self.load()
                runner.write_json(self.manifest_path, self.manifest)

    def test_each_b_trial_reuses_its_own_real_source_plan_and_recovery_metadata(self):
        context = self.load()
        with patch.object(compatibility, "MANIFEST_PATH", self.manifest_path):
            for trial, source in ((1, "v7"), (2, "v9")):
                with self.subTest(trial=trial):
                    book, lineage = runner.b_checkpoint(self.output, trial, self.plan)
                    origin = lineage["origin"]
                    self.assertEqual(book, "## OTHERS\n")
                    self.assertEqual(origin["source_plan_id"], source)
                    self.assertEqual(origin["source_plan_sha256"], context["origin_plans"][source]["plan_sha256"])
                    self.assertEqual(origin["source_plan_fingerprint"], lineage["original_fingerprint"])
                    self.assertEqual(origin["original_status"], "complete")
        historic = read_json(self.output / "B/02/result.json")
        self.assertEqual(historic["recovery"]["transport_timeout_override"], 180)
        self.assertEqual(self.original_bytes["B/02/result.json"], (self.output / "B/02/result.json").read_bytes())

    def test_failed_history_is_content_bound_but_not_a_completed_origin(self):
        context = self.load()
        self.assertEqual(context["preserved_results"]["A/02"]["original_status"], "failed")
        for trial in (2, 3):
            with self.subTest(trial=trial), self.assertRaisesRegex(ValueError, "only completed"):
                self.origin("A", trial, context)
        altered = deepcopy(self.manifest)
        altered["preserved_results"]["A/02"]["original_status"] = "complete"
        runner.write_json(self.manifest_path, altered)
        with self.assertRaisesRegex(ValueError, "status changed"):
            self.load()

    def test_v9_plan_bytes_identity_and_self_digest_are_strictly_bound(self):
        path = self.output / "plan.v9.json"
        altered = deepcopy(self.v9_plan)
        altered["config"]["llm"]["timeout"] = 180
        runner.write_json(path, altered)
        with self.assertRaisesRegex(ValueError, "plan bytes changed"):
            self.load()
        manifest = deepcopy(self.manifest)
        manifest["origin_plans"]["v9"]["plan_sha256"] = sha256(path)
        runner.write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "self-consistent"):
            self.load()

    def test_rebinding_v9_source_configuration_cannot_change_b_training(self):
        altered = deepcopy(self.v9_plan)
        altered["config"]["groups"]["B"]["max_num_rounds"] = 1
        altered["fingerprint"] = runner.digest({"config": altered["config"], "data": altered["data"],
                                               "source": altered["source_sha256"], "mock": True})
        runner.write_json(self.output / "plan.v9.json", altered)
        manifest = deepcopy(self.manifest)
        manifest["origin_plans"]["v9"].update(plan_sha256=sha256(self.output / "plan.v9.json"),
                                              fingerprint=altered["fingerprint"])
        runner.write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "only removing old C"):
            self.load()

    def test_each_preserved_result_requires_its_actual_source_plan(self):
        manifest = deepcopy(self.manifest)
        manifest["preserved_results"]["B/02"]["origin_id"] = "v7"
        runner.write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "fingerprint must match"):
            self.load()

    def test_failed_trials_remain_excluded_from_completed_summary(self):
        from tests.test_runner import offline_guards
        with offline_guards(), patch.object(compatibility, "MANIFEST_PATH", self.manifest_path):
            runner.aggregate(self.output, self.plan)
        import csv
        with (self.output / "summary.v10.csv").open(newline="") as stream:
            rows = {row["group"]: row for row in csv.DictReader(stream)}
        self.assertEqual(int(rows["A"]["completed_trials"]), 1)
        self.assertEqual(int(rows["B"]["completed_trials"]), 2)
        self.assertEqual(self.original_bytes, {name: (self.output / name).read_bytes()
                                              for name in self.original_bytes})


if __name__ == "__main__":
    unittest.main()
