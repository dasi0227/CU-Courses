from contextlib import ExitStack, redirect_stdout
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.experiments.runtime import Runtime


MODEL = 'siliconflow/deepseek-v3.1-terminus'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'bailian-unit-key',
            'DASHSCOPE_API_KEY': 'unused-dashscope-unit-key',
            'SAMBANOVA_API_KEY': 'unused-sambanova-unit-key',
        }, clear=True))
        self.stack.enter_context(patch('dotenv.load_dotenv', return_value=False))
        self.guards = [self.stack.enter_context(patch(
            target, side_effect=AssertionError('Runtime test attempted network access')))
            for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                           'socket.create_connection', 'socket.getaddrinfo',
                           'urllib.request.OpenerDirector.open',
                           'http.client.HTTPConnection.connect', 'http.client.HTTPSConnection.connect')]
        self.addCleanup(self.assert_no_network)

    def assert_no_network(self):
        for guard in self.guards:
            guard.assert_not_called()

    def config(self):
        return {
            'api_provider': 'dashscope', 'model': MODEL, 'base_url': BASE_URL,
            'api_key_env': 'DEEPSEEK_API_KEY', 'enable_thinking': False,
            'region': 'cn-beijing', 'temperature': 0, 'max_tokens': 4096,
            'retries': 2, 'sleep_seconds': 0, 'timeout': 60,
        }

    def response(self):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='2.0'))],
                               usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2), model=MODEL)

    def test_success_is_never_retried_when_local_logging_fails(self):
        client = Mock()
        client.chat.completions.create.return_value = self.response()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(self.config(), Path(tmp))
            with patch('pathlib.Path.open', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    runtime.call(client, 'dashscope', MODEL, 'prompt', 'generator', 'id', log_dir=tmp)
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(len(runtime.records), 1)
            self.assertTrue(runtime.records[0]['api_success'])

    def test_transport_writes_one_raw_jsonl_log_without_legacy_per_call_files(self):
        client = Mock()
        client.chat.completions.create.return_value = self.response()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runtime = Runtime(self.config(), output)
            runtime.stage = 'test'
            runtime.call(client, 'dashscope', MODEL, 'prompt', 'generator', 'id',
                         log_dir=str(output / 'evaluation' / 'test'))
            self.assertEqual({str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()},
                             {'llm_calls.jsonl'})
            rows = [json.loads(line) for line in (output / 'llm_calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['stage'], rows[0]['call_id'], rows[0]['response']), ('test', 'id', '2.0'))
            self.assertFalse((output / 'evaluation').exists())

    def test_processing_errors_are_persisted_but_do_not_inflate_paid_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(self.config(), Path(tmp), mock=True)
            runtime.stage = 'training'
            runtime.call(None, 'dashscope', MODEL, 'prompt', 'generator', 'train-0')
            before = runtime.summary()
            runtime.record_error(kind='parsing', role='generator', call_id='train-0',
                                 error_type='ValueError', message='invalid generated JSON',
                                 sample_id='train:0')
            after = runtime.summary()
            self.assertEqual(before, after)
            self.assertEqual(len(runtime.records), 1)
            self.assertEqual(len(runtime.errors), 1)
            events = [json.loads(line) for line in (Path(tmp) / 'llm_calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 2)
            failure = events[-1]
            self.assertEqual(failure['event'], 'processing_error')
            self.assertEqual((failure['stage'], failure['role'], failure['call_id']),
                             ('training', 'generator', 'train-0'))
            self.assertEqual(failure['error_type'], 'ValueError')
            self.assertIn('invalid generated JSON', json.dumps(failure))
            self.assertEqual(runtime.errors[0], {key: value for key, value in failure.items() if key != 'event'})

    def test_unparseable_answer_is_audited_without_retry_or_lost_raw_response(self):
        client = Mock()
        response = self.response()
        response.choices[0].message.content = 'No numeric or structured answer here.'
        client.chat.completions.create.return_value = response
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(self.config(), Path(tmp))
            text, _ = runtime.call(client, 'dashscope', MODEL, 'prompt', 'generator', 'test_eval_0')
            self.assertEqual(text, response.choices[0].message.content)
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(runtime.summary()['successful_calls'], 1)
            self.assertEqual(runtime.summary()['attempts'], 1)
            self.assertEqual(len(runtime.errors), 1)
            self.assertEqual(runtime.errors[0]['kind'], 'answer_parse')
            events = [json.loads(line) for line in (Path(tmp) / 'llm_calls.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['response'], text)
            self.assertEqual(events[1]['event'], 'processing_error')
            self.assertEqual(events[1]['call_id'], events[0]['call_id'])

    def test_real_curator_application_failure_is_consolidated_without_changing_fallback(self):
        from src import ACE
        from src.core.curator import Curator

        book = ACE._initialize_empty_playbook(None)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runtime = Runtime(self.config(), output, mock=True)
            runtime.stage = 'training'
            curator = Curator(None, 'dashscope', MODEL)
            with runtime.installed(), patch(
                'src.core.curator.apply_curator_operations',
                side_effect=ValueError('injected operation failure'),
            ) as apply_operations, redirect_stdout(io.StringIO()):
                updated, next_id, operations, call_info = curator.curate(
                    current_playbook=book, recent_reflection='Check units.',
                    question_context='Financial calculation.', current_step=1,
                    total_samples=2, token_budget=80000, playbook_stats={},
                    call_id='train_e_1_s_1',
                    log_dir=str(output / 'training' / 'detailed_llm_logs'),
                    next_global_id=7,
                )
            apply_operations.assert_called_once()
            self.assertEqual((updated, next_id, operations), (book, 7, []))
            self.assertTrue(call_info['api_success'])
            self.assertEqual(runtime.summary()['attempts'], 1)
            self.assertEqual(runtime.summary()['successful_calls'], 1)
            self.assertEqual(len(runtime.records), 1)
            self.assertEqual(runtime.records[0]['role'], 'curator')
            self.assertEqual(len(runtime.errors), 1)
            failure = runtime.errors[0]
            self.assertEqual((failure['kind'], failure['role'], failure['stage'], failure['call_id']),
                             ('curator', 'curator', 'training', 'train_e_1_s_1'))
            self.assertIn('injected operation failure', failure['message'])
            events = [json.loads(line) for line in (output / 'llm_calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[1]['event'], 'processing_error')
            self.assertIn('injected operation failure', events[1]['message'])
            self.assertEqual(events[1]['call_id'], events[0]['call_id'])
            self.assertEqual({str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()},
                             {'llm_calls.jsonl'})
            self.assertFalse((output / 'training').exists())

    def test_real_reflector_tag_parse_failure_is_consolidated_without_changing_fallback(self):
        from src.core.reflector import Reflector

        malformed = '{"reasoning": "Check units.", "bullet_tags": [oops]}'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runtime = Runtime(self.config(), output, mock=True)
            runtime.stage = 'training'
            reflector = Reflector(None, 'dashscope', MODEL)
            with runtime.installed(), patch.object(
                runtime, '_mock_response', return_value=malformed,
            ), redirect_stdout(io.StringIO()):
                response, tags, call_info = reflector.reflect(
                    question='What is the return?', reasoning_trace='A calculation.',
                    predicted_answer='0', ground_truth='1',
                    environment_feedback='Incorrect answer.', bullets_used='(empty)',
                    call_id='train_e_1_s_1_round_0',
                    log_dir=str(output / 'training' / 'detailed_llm_logs'),
                )
            self.assertEqual((response, tags), (malformed, []))
            self.assertTrue(call_info['api_success'])
            self.assertEqual(runtime.summary()['attempts'], 1)
            self.assertEqual(runtime.summary()['successful_calls'], 1)
            self.assertEqual(len(runtime.records), 1)
            self.assertEqual(runtime.records[0]['role'], 'reflector')
            self.assertEqual(len(runtime.errors), 1)
            failure = runtime.errors[0]
            self.assertEqual((failure['kind'], failure['role'], failure['stage'], failure['call_id']),
                             ('reflector_parse', 'reflector', 'training', 'train_e_1_s_1_round_0'))
            self.assertTrue(failure['message'])
            events = [json.loads(line) for line in (output / 'llm_calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]['response'], malformed)
            self.assertEqual(events[1]['event'], 'processing_error')
            self.assertEqual(events[1]['kind'], 'reflector_parse')
            self.assertEqual(events[1]['call_id'], events[0]['call_id'])
            self.assertEqual({str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()},
                             {'llm_calls.jsonl'})
            self.assertFalse((output / 'training').exists())

    def test_retries_are_bounded_and_each_attempt_is_audited(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError('offline timeout')
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(self.config(), Path(tmp))
            with self.assertRaises(RuntimeError):
                runtime.call(client, 'dashscope', MODEL, 'prompt', 'generator', 'id')
            self.assertEqual(client.chat.completions.create.call_count, 3)
            self.assertEqual(runtime.summary()['failed_attempts'], 3)

    def test_authentication_or_bad_request_does_not_retry(self):
        class AuthError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
        for status in (400, 401):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                client = Mock()
                client.chat.completions.create.side_effect = AuthError(status)
                runtime = Runtime(self.config(), Path(tmp))
                with self.assertRaises(RuntimeError):
                    runtime.call(client, 'dashscope', MODEL, 'prompt', 'generator', 'id')
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_sdk_parameters_disable_thinking_for_every_role_with_uniform_budget(self):
        client = Mock()
        replies = [self.response() for _ in range(3)]
        replies[2].choices[0].message.content = json.dumps({'reasoning': 'offline', 'operations': []})
        client.chat.completions.create.side_effect = replies
        with tempfile.TemporaryDirectory() as tmp, patch('openai.OpenAI', return_value=client) as sdk:
            runtime = Runtime(self.config(), Path(tmp))
            self.assertEqual(runtime.clients('dashscope'), (client, client, client))
            constructor = sdk.call_args.kwargs
            self.assertEqual(constructor['api_key'], 'bailian-unit-key')
            self.assertEqual(constructor['base_url'], BASE_URL)
            self.assertEqual(constructor['timeout'], 60)
            self.assertEqual(constructor['max_retries'], 0)
            for role in ('generator', 'reflector', 'curator'):
                runtime.call(client, 'dashscope', MODEL, role, role, role, max_tokens=4096)
            calls = client.chat.completions.create.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertEqual([call.kwargs['max_tokens'] for call in calls], [4096, 4096, 4096])
            for call in calls:
                self.assertEqual(call.kwargs['extra_body'], {'enable_thinking': False})
                self.assertEqual(call.kwargs['model'], MODEL)
                self.assertEqual(call.kwargs['temperature'], 0)
                self.assertNotIn('api_key', call.kwargs)
            for row in runtime.records:
                self.assertEqual(row['api_provider'], 'dashscope')
                self.assertEqual(row['base_url'], BASE_URL)
                self.assertIs(row['enable_thinking'], False)
                self.assertEqual(row['response_model'], MODEL)
            audit = (Path(tmp) / 'llm_calls.jsonl').read_text()
            for secret in ('bailian-unit-key', 'unused-dashscope-unit-key', 'unused-sambanova-unit-key'):
                self.assertNotIn(secret, audit)

    def test_missing_deepseek_variable_does_not_use_other_provider_keys(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'SAMBANOVA_API_KEY': 'unused-sambanova-unit-key',
            'DASHSCOPE_API_KEY': 'unused-dashscope-unit-key',
        }, clear=True), patch('openai.OpenAI') as sdk:
            runtime = Runtime(self.config(), Path(tmp))
            with self.assertRaises((KeyError, ValueError)):
                runtime.clients('dashscope')
            sdk.assert_not_called()

    def test_runtime_rejects_invalid_configuration_before_client_creation(self):
        cases = ({'api_provider': 'sambanova'}, {'model': 'deepseek-v3.1'},
                 {'api_key_env': 'DASHSCOPE_API_KEY'}, {'enable_thinking': True},
                 {'base_url': 'https://api.deepseek.com/v1'})
        with tempfile.TemporaryDirectory() as tmp, patch('openai.OpenAI') as sdk:
            for changed in cases:
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    Runtime({**self.config(), **changed}, Path(tmp))
            sdk.assert_not_called()


if __name__ == '__main__':
    unittest.main()
