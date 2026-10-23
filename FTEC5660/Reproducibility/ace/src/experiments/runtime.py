"""Bounded, audited LLM transport for the three experiment arms.

The original role prompts and ACE training loop are retained. Transport is
injected to apply one audited API configuration to every role.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib
import json
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

from .data import validate_llm_config


class Runtime:
    def __init__(self, config, output_dir: Path):
        validate_llm_config(config)
        self.config = config
        self.output_dir = Path(output_dir)
        self.stage = 'setup'
        self.records = []
        self.errors = []
        self._lock = threading.Lock()
        self._client = None

    def clients(self, api_provider):
        if api_provider != self.config['api_provider']:
            raise ValueError('Client provider must match the configured Bailian provider')
        if self._client is None:
            from openai import OpenAI
            secret = os.environ.get(self.config['api_key_env'], '').strip()
            if not secret:
                raise ValueError(f"Missing key: {self.config['api_key_env']}")
            self._client = OpenAI(
                api_key=secret,
                base_url=self.config['base_url'],
                timeout=self.config.get('timeout', 60), max_retries=0,
            )
        return (self._client,) * 3

    def call(self, client, api_provider, model, prompt, role, call_id,
             max_tokens=4096, log_dir=None, use_json_mode=False, **_):
        if api_provider != self.config['api_provider'] or model != self.config['model']:
            raise ValueError('Every role must use the configured Bailian provider and model')
        if use_json_mode:
            raise ValueError('This Bailian model uses JSON prompts, without response_format=json_object')
        start = time.monotonic()
        params = dict(model=model, messages=[{'role': 'user', 'content': prompt}],
                      temperature=self.config['temperature'], max_tokens=max_tokens,
                      extra_body={'enable_thinking': False})
        attempts = self.config.get('retries', 2) + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            info = {'role': role, 'call_id': call_id, 'stage': self.stage,
                    'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
                    'temperature': self.config['temperature'], 'attempt': attempt,
                    'api_provider': api_provider, 'base_url': self.config['base_url'],
                    'enable_thinking': False,
                    'mock': False}
            try:
                response = client.chat.completions.create(**params)
                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise RuntimeError('Empty model response')
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', None)
                completion_tokens = getattr(usage, 'completion_tokens', None)
                response_model = getattr(response, 'model', None)
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                info.update(response=text, prompt_num_tokens=prompt_tokens,
                            response_num_tokens=completion_tokens, api_success=True,
                            response_model=response_model, finish_reason=finish_reason,
                            total_time=time.monotonic() - start)
            except Exception as exc:
                last_error = exc
                info.update(api_success=False, error=type(exc).__name__, error_message=str(exc),
                            total_time=time.monotonic() - start)
                self._record(info)
                status = getattr(exc, 'status_code', None)
                # Configuration/authentication errors cannot be repaired by retry.
                if status in (400, 401, 403, 404, 422) or attempt == attempts:
                    break
                time.sleep(self.config.get('sleep_seconds', 1))
                continue
            # An already paid request must never be retried because disk logging
            # or output validation failed. Such failures stop the trial instead.
            self._record(info)
            if role == 'generator':
                from src.common.utils import extract_answer, _NO_ANSWER
                if extract_answer(text) == _NO_ANSWER:
                    self.record_error(kind='answer_parse', role=role, call_id=call_id,
                                      error_type='AnswerParseError',
                                      message='The model response has no extractable final answer')
            if role == 'curator':
                from src.core.curator import Curator
                try:
                    Curator(None, api_provider, model)._extract_and_validate_operations(text)
                except Exception as exc:
                    self.record_error(kind='curator_schema', role=role, call_id=call_id,
                                      error_type=type(exc).__name__, message=str(exc))
                    raise
            return text, info
        raise RuntimeError(f'{role} call failed after bounded attempts ({type(last_error).__name__})') from None

    def _record(self, info):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        info = self._safe(info)
        with self._lock:
            self.records.append(dict(info))
            if not info['api_success']:
                self.errors.append({
                    'kind': 'api', 'role': info['role'], 'stage': info['stage'],
                    'call_id': info['call_id'], 'attempt': info['attempt'],
                    'error_type': info['error'], 'message': info.get('error_message', info['error']),
                })
            with (self.output_dir / 'llm_calls.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(info, ensure_ascii=False) + '\n')

    def _safe(self, value):
        if isinstance(value, str):
            for name in (self.config['api_key_env'], 'TYPESAFE_API_KEY'):
                secret = os.environ.get(name, '')
                if secret:
                    value = value.replace(secret, '[REDACTED]')
            return value
        if isinstance(value, dict):
            return {key: self._safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._safe(item) for item in value]
        return value

    def record_error(self, *, kind, role, call_id, error_type, message, stage=None, **details):
        """Persist processing failures without counting them as paid requests."""
        event = self._safe({'kind': kind, 'role': role, 'call_id': call_id,
                            'stage': self.stage if stage is None else stage,
                            'error_type': error_type, 'message': message, **details})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.errors.append(event)
            with (self.output_dir / 'llm_calls.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'event': 'processing_error', **event}, ensure_ascii=False) + '\n')

    def _curator_failure(self, save_path, step, failure_type, curator_response, epoch,
                         error_details=None):
        with self._lock:
            previous = next((row for row in reversed(self.records)
                             if row['role'] == 'curator' and row['stage'] == self.stage), {})
        self.record_error(kind='curator', role='curator',
                          call_id=previous.get('call_id', f'curator_step_{step}'),
                          error_type=failure_type, message=error_details or failure_type,
                          step=step)

    def summary(self):
        successful = [r for r in self.records if r['api_success']]
        by_stage = {}
        for stage in sorted({r['stage'] for r in self.records}):
            rows = [r for r in self.records if r['stage'] == stage]
            good = [r for r in rows if r['api_success']]
            by_stage[stage] = {
                'successful_calls': len(good), 'attempts': len(rows),
                'prompt_tokens': sum(r.get('prompt_num_tokens') or 0 for r in good),
                'completion_tokens': sum(r.get('response_num_tokens') or 0 for r in good),
            }
        return {
            'attempts': len(self.records), 'successful_calls': len(successful),
            'failed_attempts': len(self.records) - len(successful),
            'prompt_tokens': sum(r.get('prompt_num_tokens') or 0 for r in successful),
            'completion_tokens': sum(r.get('response_num_tokens') or 0 for r in successful),
            'usage_missing_calls': sum(r.get('prompt_num_tokens') is None or
                                       r.get('response_num_tokens') is None for r in successful),
            'mock': False,
            'stages': {stage: sum(r['stage'] == stage for r in self.records)
                       for stage in sorted({r['stage'] for r in self.records})},
            'stage_usage': by_stage,
        }

    @contextmanager
    def installed(self):
        """Inject the audited API clients, transport, and error handlers."""
        ace_module = importlib.import_module('src.ace')
        with ExitStack() as stack:
            stack.enter_context(patch.object(ace_module, 'initialize_clients', self.clients))
            for name in ('generator', 'reflector', 'curator'):
                module = importlib.import_module(f'src.core.{name}')
                stack.enter_context(patch.object(module, 'timed_llm_call', self.call))
            curator_module = importlib.import_module('src.core.curator')
            stack.enter_context(patch.object(curator_module, 'log_curator_operation_diff', lambda *args, **kwargs: None))
            stack.enter_context(patch.object(curator_module, 'log_curator_failure', self._curator_failure))
            reflector_module = importlib.import_module('src.core.reflector')
            stack.enter_context(patch.object(
                reflector_module.Reflector, '_report_parse_failure',
                lambda reflector, error, call_id: self.record_error(
                    kind='reflector_parse', role='reflector', call_id=call_id,
                    error_type=type(error).__name__, message=str(error)),
            ))
            yield
