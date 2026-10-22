"""Legacy CLI provider checks using fake credentials and fake clients only."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


with patch("dotenv.load_dotenv", return_value=False):
    utils = importlib.import_module("src.common.utils")
    llm = importlib.import_module("src.common.llm")
    finance = importlib.import_module("eval.finance.run")


MODEL = "siliconflow/deepseek-v3.1-terminus"
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def fake_response(model=MODEL):
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"final_answer":"2.0"}'))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )
    if model is not None:
        result.model = model
    return result


class LegacyProviderTests(unittest.TestCase):
    def test_dashscope_key_and_endpoint_are_distinct_from_sambanova(self):
        for override in (None, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"):
            env = {"DEEPSEEK_API_KEY": "offline-dashscope", "SAMBANOVA_API_KEY": "offline-samba"}
            if override:
                env["DASHSCOPE_BASE_URL"] = override
            with self.subTest(override=override), patch.dict(os.environ, env, clear=True):
                with patch.object(utils.openai, "OpenAI") as constructor:
                    self.assertEqual(len(utils.initialize_clients("dashscope")), 3)
                    self.assertEqual(constructor.call_count, 3)
                    for call in constructor.call_args_list:
                        self.assertEqual(call.kwargs, {"api_key": "offline-dashscope", "base_url": override or DEFAULT_ENDPOINT})
                    utils.initialize_clients("sambanova")
                    self.assertEqual(constructor.call_args.kwargs["api_key"], "offline-samba")
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "offline-samba"}, clear=True):
            with patch.object(utils.openai, "OpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                    utils.initialize_clients("dashscope")
                constructor.assert_not_called()

    def test_dashscope_disables_thinking_for_each_role_and_records_model(self):
        client = Mock()
        client.chat.completions.create.return_value = fake_response()
        with patch.object(llm, "log_llm_call") as log:
            for role in ("generator", "reflector", "curator"):
                _, info = llm.timed_llm_call(client, "dashscope", MODEL, "offline prompt", role, f"test_{role}", log_dir="offline-placeholder")
                params = client.chat.completions.create.call_args.kwargs
                self.assertEqual(params["extra_body"], {"enable_thinking": False})
                self.assertEqual(params["model"], MODEL)
                self.assertEqual(params["max_tokens"], 4096)
                self.assertNotIn("response_format", params)
                self.assertEqual(info["api_provider"], "dashscope")
                self.assertFalse(info["enable_thinking"])
                self.assertEqual(info["response_model"], MODEL)
            self.assertEqual(log.call_count, 3)
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_missing_response_model_is_recorded_as_none(self):
        client = Mock()
        client.chat.completions.create.return_value = fake_response(model=None)
        _, info = llm.timed_llm_call(client, "dashscope", MODEL, "offline prompt", "generator", "test_missing_model")
        self.assertIsNone(info["response_model"])

    def test_non_dashscope_json_mode_remains_supported(self):
        client = Mock()
        client.chat.completions.create.return_value = fake_response(model="DeepSeek-V3.1")
        llm.timed_llm_call(client, "sambanova", "DeepSeek-V3.1", "offline prompt", "generator", "test_samba", use_json_mode=True)
        params = client.chat.completions.create.call_args.kwargs
        self.assertEqual(params["response_format"], {"type": "json_object"})
        self.assertNotIn("extra_body", params)

    def test_cli_defaults_and_explicit_legacy_provider(self):
        argv = ["finance", "--task_name", "formula", "--save_path", "offline-placeholder"]
        with patch("sys.argv", argv):
            args = finance.parse_args()
        self.assertEqual(args.api_provider, "dashscope")
        self.assertEqual((args.generator_model, args.reflector_model, args.curator_model), (MODEL,) * 3)
        with patch("sys.argv", argv + ["--api_provider", "sambanova"]):
            self.assertEqual(finance.parse_args().api_provider, "sambanova")

    def test_unsupported_dashscope_json_mode_fails_before_client_or_request(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "does not support API JSON mode"):
            llm.timed_llm_call(client, "dashscope", MODEL, "offline prompt", "generator", "test_json", use_json_mode=True)
        client.chat.completions.create.assert_not_called()
        with patch("sys.argv", ["finance", "--task_name", "formula", "--save_path", "offline-placeholder", "--json_mode"]):
            with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as exit_status:
                finance.parse_args()
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("omit --json_mode", stderr.getvalue())

    def test_cli_help_exits_without_constructing_any_client(self):
        with patch("sys.argv", ["finance", "--help"]), patch.object(utils.openai, "OpenAI") as constructor:
            with redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit) as exit_status:
                finance.main()
            self.assertEqual(exit_status.exception.code, 0)
            self.assertIn("dashscope", stdout.getvalue())
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
