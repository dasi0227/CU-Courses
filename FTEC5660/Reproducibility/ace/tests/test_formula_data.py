"""Standard-library checks for Formula provenance, sampling, and configuration."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Loading this one module directly keeps the data audit independent of model SDKs.
_spec = importlib.util.spec_from_file_location("formula_experiment_data", PROJECT_ROOT / "src/experiments/data.py")
data_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_module)


class FormulaDataTests(unittest.TestCase):
    def config(self, name="small"):
        return data_module.load_config(PROJECT_ROOT / "configs/formula" / f"{name}.json")

    def test_committed_profiles_have_expected_counts_and_original_inputs(self):
        for profile, counts in (("small", (100, 50, 200)), ("paper", (500, 300, 200)), ("smoke", (2, 2, 2))):
            with self.subTest(profile=profile):
                prepared = data_module.prepare_data(self.config(profile), PROJECT_ROOT)
                self.assertEqual(tuple(len(prepared[split]) for split in data_module.SPLITS), counts)
                self.assertEqual(prepared["manifest"]["formal_metrics"], profile != "smoke")
                for split in data_module.SPLITS:
                    self.assertTrue(all(row["raw_context"] == row["context"] for row in prepared[split]))
                    self.assertTrue(all("Formula:" in row["context"] and "Explanation:" in row["context"] for row in prepared[split]))

    def test_every_profile_uses_the_same_bailian_model_and_unchanged_typesafe_endpoint(self):
        for profile in ("small", "paper", "smoke"):
            with self.subTest(profile=profile):
                config = self.config(profile)
                llm = config["llm"]
                self.assertEqual(llm["api_provider"], "dashscope")
                self.assertEqual(llm["model"], "siliconflow/deepseek-v3.1-terminus")
                self.assertEqual(llm["api_key_env"], "DEEPSEEK_API_KEY")
                self.assertEqual(llm["region"], "cn-beijing")
                self.assertIs(llm["enable_thinking"], False)
                self.assertEqual(config["jev"]["api_key_env"], "TYPESAFE_API_KEY")
                self.assertEqual(config["jev"]["model"], "jev-1.13.0")
                self.assertEqual(config["jev"]["base_url"], "https://api.typesafe.ai/v1/systemone")
                self.assertEqual(config["jev"]["k"], 10)
                self.assertNotIn("k_candidates", config["jev"])

    def test_bailian_shared_and_beijing_workspace_endpoints_are_valid(self):
        for endpoint in ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                         "https://example-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"):
            with self.subTest(endpoint=endpoint):
                config = self.config()
                config["llm"]["base_url"] = endpoint
                data_module.validate_llm_config(config["llm"])
                data_module.validate_config(config)

    def test_profiles_define_only_the_three_controlled_experiments(self):
        self.assertEqual(data_module.GROUPS, ("A", "B", "C"))
        for profile in ("small", "paper", "smoke"):
            with self.subTest(profile=profile):
                config = self.config(profile)
                self.assertEqual(tuple(config["groups"]), data_module.GROUPS)
                self.assertEqual(config["groups"]["A"], {"enabled": True, "kind": "empty_playbook"})
                self.assertEqual(config["groups"]["B"]["max_num_rounds"], 3)
                self.assertEqual(config["groups"]["C"], {"enabled": True, "kind": "jev_filter", "depends_on": "B"})
                self.assertNotIn("raw_max_tokens", config["groups"]["A"])
                self.assertNotIn("novelty_threshold", config["jev"])
                self.assertNotIn("reusability_threshold", config["jev"])

    def test_small_profile_is_balanced_and_reproducible(self):
        config = self.config()
        first = data_module.prepare_data(config, PROJECT_ROOT)
        second = data_module.prepare_data(config, PROJECT_ROOT)
        self.assertEqual(first, second)
        for split, count in (("train", 2), ("val", 1), ("test", 4)):
            frequencies = Counter(row["formula_name"] for row in first[split])
            self.assertEqual(len(frequencies), 50)
            self.assertEqual(set(frequencies.values()), {count})
        config["seed"] += 1
        changed = data_module.prepare_data(config, PROJECT_ROOT)
        self.assertNotEqual([row["id"] for row in first["train"]], [row["id"] for row in changed["train"]])
        self.assertEqual(first["test"], changed["test"])

    def test_manifest_reconstructs_samples_from_source_lines(self):
        prepared = data_module.prepare_data(self.config(), PROJECT_ROOT)
        for split in data_module.SPLITS:
            provenance = prepared["manifest"]["sources"][split]
            content = Path(provenance["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), provenance["sha256"])
            source_lines = content.decode("utf-8").splitlines()
            self.assertEqual(provenance["selected_count"], len(prepared[split]))
            for row in prepared[split]:
                original = json.loads(source_lines[row["source_line"] - 1])
                self.assertEqual(row["context"], original["context"])
                self.assertEqual(row["target"], original["target"])
                self.assertEqual(row["id"], f"{split}:{row['source_line']:06d}")

    def test_relative_sources_use_project_root_and_physical_line_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = "Use formula Example to answer the question. Formula: x = a / b, Explanation: division. Question: example. Answer:"
            records = [{"context": context, "target": 2.0}, {"context": context, "target": "3.00"}]
            source = "\n" + json.dumps(records[0]) + "\n\n" + json.dumps(records[1]) + "\n"
            (root / "fixture.jsonl").write_text(source, encoding="utf-8")
            config = self.config("smoke")
            for split in data_module.SPLITS:
                config["data"][f"{split}_path"] = "fixture.jsonl"
            prepared = data_module.prepare_data(config, root)
            self.assertEqual([row["source_line"] for row in prepared["train"]], [2, 4])
            self.assertEqual([row["id"] for row in prepared["test"]], ["test:000002", "test:000004"])
            self.assertIsInstance(prepared["train"][0]["target"], float)
            self.assertEqual(prepared["train"][1]["target"], "3.00")
            self.assertEqual(set(root.iterdir()), {root / "fixture.jsonl"})

    def test_invalid_settings_fail_before_loading_data(self):
        cases = (
            ("repeats below assignment minimum", lambda c: c.update(repeats=2)),
            ("boolean is not integer seed", lambda c: c.update(seed=True)),
            ("non-positive k", lambda c: c["jev"].update(k=0)),
            ("k must remain fixed at ten", lambda c: c["jev"].update(k=4)),
            ("old top-five k is rejected", lambda c: c["jev"].update(k=5)),
            ("old k tuning is rejected", lambda c: c["jev"].update(k_candidates=[3, 5, 10])),
            ("single k candidate is still tuning configuration", lambda c: c["jev"].update(k_candidates=[10])),
            ("negative retry count", lambda c: c["llm"].update(retries=-1)),
            ("zero request timeout", lambda c: c["llm"].update(timeout=0)),
            ("wrong provider", lambda c: c["llm"].update(api_provider="sambanova")),
            ("wrong model version", lambda c: c["llm"].update(model="deepseek-v3.1")),
            ("wrong key environment", lambda c: c["llm"].update(api_key_env="DASHSCOPE_API_KEY")),
            ("thinking enabled", lambda c: c["llm"].update(enable_thinking=True)),
            ("thinking string", lambda c: c["llm"].update(enable_thinking="false")),
            ("wrong region", lambda c: c["llm"].update(region="cn-shanghai")),
            ("plaintext endpoint", lambda c: c["llm"].update(base_url="http://dashscope.aliyuncs.com/compatible-mode/v1")),
            ("unrelated endpoint", lambda c: c["llm"].update(base_url="https://api.deepseek.com/v1")),
            ("international endpoint", lambda c: c["llm"].update(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")),
            ("wrong endpoint path", lambda c: c["llm"].update(base_url="https://dashscope.aliyuncs.com/v1")),
            ("endpoint suffix spoof", lambda c: c["llm"].update(base_url="https://dashscope.aliyuncs.com.evil.example/compatible-mode/v1")),
            ("wrong workspace region", lambda c: c["llm"].update(base_url="https://example.cn-shanghai.maas.aliyuncs.com/compatible-mode/v1")),
            ("old B five-round cap", lambda c: c["groups"]["B"].update(max_num_rounds=5)),
            ("removed one-round training arm", lambda c: c["groups"].update(C={"enabled": True, "kind": "ace", "max_num_rounds": 1})),
            ("incorrect baseline kind", lambda c: c["groups"]["A"].update(kind="ace")),
            ("removed E group", lambda c: c["groups"].update(E={"enabled": True})),
            ("removed F group", lambda c: c["groups"].update(F={"enabled": True})),
            ("unknown group", lambda c: c["groups"].update(G={"enabled": True})),
            ("incorrect C dependency", lambda c: c["groups"]["C"].update(depends_on="A")),
            ("removed D group", lambda c: c["groups"].update(D={"enabled": True})),
            ("batch workflow changes", lambda c: c["training"].update(batch_size=2)),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                config = deepcopy(self.config())
                mutate(config)
                with self.assertRaises(ValueError):
                    data_module.validate_config(config)

    def test_insufficient_formula_samples_are_rejected(self):
        rows = [{"source_line": 1, "formula_name": "One"}]
        with self.assertRaisesRegex(ValueError, "fewer than 2"):
            data_module._sample_per_formula(rows, 2, 42)


if __name__ == "__main__":
    unittest.main()
