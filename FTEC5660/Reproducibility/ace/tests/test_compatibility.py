"""Offline checks that ABC history compatibility is explicit and content bound."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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
        preserved = {}
        for group in "AB":
            prefix = self.output / group / "01"
            prefix.mkdir(parents=True)
            result = {
                "group": group, "trial": 1, "status": "complete", "mock": True,
                "formal_metrics": False, "fingerprint": self.old_plan["fingerprint"],
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
                "original_fingerprint": result["fingerprint"],
                "files_sha256": {name: sha256(prefix / name) for name in names},
            }
        changes = {
            name: {"old_sha256": old_source.get(name), "new_sha256": self.plan["source_sha256"].get(name)}
            for name in set(old_source) | set(self.plan["source_sha256"])
            if old_source.get(name) != self.plan["source_sha256"].get(name)
        }
        self.manifest = {
            "schema_version": 1,
            "migration_id": "offline-v7-to-v8-fixture",
            "legacy": {"schema_version": 7, "protocol": self.old_plan["protocol"],
                       "fingerprint": self.old_plan["fingerprint"],
                       "plan_sha256": sha256(self.output / "plan.json")},
            "target": {key: deepcopy(self.plan[key]) for key in (
                "schema_version", "protocol", "fingerprint", "config", "data",
                "source_sha256", "mode", "formal_metrics",
            )},
            "source_changes": changes,
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
                         ("plan.v8.json", "invocations.v8.jsonl", "summary.v8.csv"))
        fresh = Path(self.temporary.name) / "fresh"
        self.assertEqual(compatibility.storage_names(fresh, self.plan, None),
                         ("plan.json", "invocations.jsonl", "summary.csv"))

    def test_each_historical_file_is_content_bound_including_predictions_and_best_book(self):
        for relative, before in self.original_bytes.items():
            if relative == "plan.json":
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
        manifest["legacy"]["plan_sha256"] = sha256(self.output / "plan.json")
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
        for group, trial in (("A", 2), ("B", 2), ("C", 1), ("D", 1)):
            with self.subTest(group=group, trial=trial):
                result = {**original, "group": group, "trial": trial}
                runner.write_json(self.output / group / f"{trial:02d}" / "result.json", result)
                with self.assertRaises(ValueError):
                    compatibility.validate_result_origin(
                        self.output, group, trial, self.plan, result, compatibility=context,
                    )

    def test_manifest_must_describe_exactly_the_approved_config_and_source_transition(self):
        mutations = (
            lambda m: m["target"]["config"]["groups"]["B"].update(max_num_rounds=1),
            lambda m: m["source_changes"].clear(),
            lambda m: m["legacy"].update(fingerprint="0" * 64),
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


if __name__ == "__main__":
    unittest.main()
