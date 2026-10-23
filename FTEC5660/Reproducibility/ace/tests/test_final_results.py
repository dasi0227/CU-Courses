"""Read-only final-view invariants, using small synthetic artifacts only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import validate_formula_results as validator


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def embedded_document(context, path, document):
    text = json.dumps(document, sort_keys=True, ensure_ascii=False) + '\n'
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    context['documents'][sha] = {'original_path': str(path), 'json_text': text}
    return sha


def historical_plan(config):
    plan = {'config': config, 'data': {}, 'source_sha256': {}, 'mode': 'mock'}
    plan['fingerprint'] = validator.digest({'config': config, 'data': {}, 'source': {}, 'mock': True})
    return plan


class FinalResultValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'A' / '02'
        self.folder.mkdir(parents=True)
        self.samples = [{'id': 'test:000001', 'target': '1000.0'},
                        {'id': 'test:000002', 'target': '0.0'}]
        self.config = {'model': 'example', 'api_provider': 'dashscope', 'base_url': 'https://example.invalid',
                       'temperature': 0, 'max_tokens': 4096, 'enable_thinking': False}
        self.predictions = [
            {'index': 0, 'sample_id': self.samples[0]['id'], 'call_id': 'test_eval_0', 'stage': 'test',
             'target': '1000.0', 'prediction': '1,000', 'is_correct': True, 'api_success': True, 'error': None},
            {'index': 1, 'sample_id': self.samples[1]['id'], 'call_id': 'test_eval_1', 'stage': 'test',
             'target': '0.0', 'prediction': validator.NO_ANSWER, 'is_correct': False, 'api_success': True,
             'error': 'AnswerParseError: no extractable final answer'},
        ]
        self.calls = [dict(self.config, role='generator', stage='test', call_id=f'test_eval_{index}',
                           api_success=True, response='raw model response', prompt='original prompt',
                           prompt_num_tokens=10, response_num_tokens=20)
                      for index in (1, 0)]  # Concurrent response completion order is intentional.
        self.result = {
            'group': 'A', 'trial': 2, 'status': 'complete', 'artifact_view': 'final_predictions',
            'test': {'accuracy': .5, 'correct': 1, 'total': 2, 'no_answer': 1,
                     'api_success_count': 2, 'api_failure_count': 0, 'success_rate': 1.0, 'stage': 'test'},
            'wall_seconds': 123.0,
            'llm': {'attempts': 4, 'successful_calls': 3, 'prompt_tokens': 30, 'completion_tokens': 90},
            'final_llm_usage': {'successful_calls': 2, 'prompt_tokens': 20, 'completion_tokens': 40},
        }
        self.save()

    def save(self):
        write_json(self.folder / 'result.json', self.result)
        write_jsonl(self.folder / 'predictions.jsonl', self.predictions)
        write_jsonl(self.folder / 'llm_calls.jsonl', self.calls)

    def validate(self):
        return validator.validate_trial(self.folder, 'A', 2, self.samples, self.config)

    def test_final_view_accepts_no_answer_and_preserves_larger_actual_usage_without_writes(self):
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        self.assertEqual(self.validate(), self.result)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})

    def test_duplicate_missing_or_reordered_prediction_ids_are_rejected(self):
        for mutate in (lambda: self.predictions.reverse(),
                       lambda: self.predictions[1].update(sample_id=self.samples[0]['id']),
                       lambda: self.predictions.pop()):
            with self.subTest(mutate=mutate):
                before = deepcopy(self.predictions)
                mutate()
                self.save()
                with self.assertRaisesRegex(ValueError, 'IDs/count/order'):
                    self.validate()
                self.predictions = before

    def test_metrics_are_recomputed_from_exact_match_predictions(self):
        for key, value in (('correct', 2), ('accuracy', 1), ('no_answer', 0), ('api_failure_count', 1)):
            with self.subTest(key=key):
                before = self.result['test'][key]
                self.result['test'][key] = value
                self.save()
                with self.assertRaisesRegex(ValueError, f'test.{key}'):
                    self.validate()
                self.result['test'][key] = before

    def test_prediction_flag_and_target_tampering_are_rejected(self):
        for key, value, error in (('is_correct', True, 'exact-match'), ('target', '-1', 'target changed')):
            with self.subTest(key=key):
                before = self.predictions[1][key]
                self.predictions[1][key] = value
                self.save()
                with self.assertRaisesRegex(ValueError, error):
                    self.validate()
                self.predictions[1][key] = before

    def test_final_tokens_must_match_retained_responses_exactly(self):
        self.result['final_llm_usage']['completion_tokens'] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'token/call totals mismatch'):
            self.validate()

    def test_actual_usage_cannot_undercount_final_responses(self):
        self.result['llm']['prompt_tokens'] = 19
        self.save()
        with self.assertRaisesRegex(ValueError, 'cumulative usage undercounts'):
            self.validate()

    def test_historical_transport_fields_and_missing_responses_are_rejected(self):
        for key, value in (('attempt', 4), ('transport_timeout_seconds', 180), ('response', '')):
            with self.subTest(key=key):
                before = deepcopy(self.calls)
                self.calls[0][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.validate()
                self.calls = before

    def test_duplicate_final_response_id_is_rejected(self):
        self.calls[0]['call_id'] = self.calls[1]['call_id']
        self.save()
        with self.assertRaisesRegex(ValueError, 'call IDs'):
            self.validate()

    def test_changed_model_budget_is_rejected(self):
        self.calls[0]['max_tokens'] = 8192
        self.save()
        with self.assertRaisesRegex(ValueError, 'changed model setting max_tokens'):
            self.validate()

    def test_clean_rules_do_not_rewrite_or_constrain_untouched_legacy_views(self):
        self.result.pop('artifact_view')
        self.result.pop('final_llm_usage')
        self.result['errors'] = [{'kind': 'legacy'}]
        self.result['recovery'] = {'original': 'kept'}
        self.calls[0]['attempt'] = 4
        self.save()
        self.assertEqual(self.validate(), self.result)

    def test_clean_view_rejects_embedded_history(self):
        self.result['recovery'] = {'original': 'removed'}
        self.save()
        with self.assertRaisesRegex(ValueError, 'historical records'):
            self.validate()

    def test_cli_imports_without_site_packages_or_credentials(self):
        completed = subprocess.run(
            [sys.executable, '-S', 'scripts/validate_formula_results.py', '--help'], cwd=ROOT,
            capture_output=True, text=True, env={'PATH': '/usr/bin:/bin'}, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.book = self.root / 'B/02/training/best_playbook.txt'
        self.book.parent.mkdir(parents=True)
        self.book.write_text('## Lessons\nKeep units consistent.\n')
        self.b_result = {'fingerprint': 'source-fingerprint', 'playbook_sha256': validator.sha256(self.book),
                         'llm': {'stage_usage': {'training': dict(successful_calls=3, attempts=4,
                                                                 prompt_tokens=10, completion_tokens=20)}}}
        write_json(self.root / 'B/02/result.json', self.b_result)
        shared = self.b_result['llm']['stage_usage']['training']
        self.c_result = {
            'playbook_sha256': self.b_result['playbook_sha256'],
            'b_checkpoint': {'group': 'B', 'trial': 2, 'path': str(self.book),
                             'sha256': self.b_result['playbook_sha256'],
                             'original_fingerprint': self.b_result['fingerprint'],
                             'origin': {'kind': 'current', 'fingerprint': self.b_result['fingerprint'],
                                        'result_sha256': validator.sha256(self.root / 'B/02/result.json')},
                             'training_cost_included_in_C_incremental_usage': False,
                             'shared_learning_excludes_B_test': True, 'shared_learning_llm_usage': shared},
            'llm': dict.fromkeys(shared, 1),
            'llm_usage_including_shared_learning': {key: value + 1 for key, value in shared.items()},
            'selection_k': 10, 'selection_primitive': 'score',
            'selection_ranking': 'score_desc_then_playbook_order',
            'selection_confidence_used_for_ranking': False,
        }

    def validate(self):
        validator.validate_checkpoint(self.root, 2, self.b_result, self.c_result)

    def test_same_number_book_and_recorded_original_result_are_verified(self):
        self.validate()

    def test_book_tampering_is_rejected(self):
        self.book.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.validate()

    def test_other_trial_and_source_fingerprint_are_rejected(self):
        for key, value, message in (('trial', 1, 'matching B'),
                                    ('original_fingerprint', 'different', 'source fingerprint')):
            with self.subTest(key=key):
                before = self.c_result['b_checkpoint'][key]
                self.c_result['b_checkpoint'][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                self.c_result['b_checkpoint'][key] = before

    def test_recorded_b_result_sha_is_not_silently_rebound(self):
        self.c_result['b_checkpoint']['origin']['result_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'result identity/SHA-256'):
            self.validate()

    def test_legacy_checkpoint_uses_embedded_manifest_and_source_plan(self):
        context = {'kind': 'consolidated_plan', 'documents': {}}
        source_path = self.root / 'plan.v9.json'
        manifest_path = self.root / 'deleted-configs/compatibility.json'
        source_plan = historical_plan({'historical': 9})
        source_sha = embedded_document(context, source_path, source_plan)
        self.b_result['fingerprint'] = source_plan['fingerprint']
        write_json(self.root / 'B/02/result.json', self.b_result)
        lineage = self.c_result['b_checkpoint']
        lineage['original_fingerprint'] = source_plan['fingerprint']
        source = {'path': 'plan.v9.json', 'plan_sha256': source_sha,
                  'fingerprint': source_plan['fingerprint']}
        result_sha = validator.sha256(self.root / 'B/02/result.json')
        manifest = {'migration_id': 'historical-migration', 'origin_plans': {'v9': source},
                    'preserved_results': {'B/02': {'original_fingerprint': source_plan['fingerprint'],
                                                   'files_sha256': {'result.json': result_sha}}}}
        manifest_sha = embedded_document(context, manifest_path, manifest)
        lineage['origin'] = {
            'kind': 'legacy_compatible', 'fingerprint': source_plan['fingerprint'],
            'result_sha256': result_sha, 'manifest_path': str(manifest_path),
            'manifest_sha256': manifest_sha, 'migration_id': manifest['migration_id'],
            'source_plan_id': 'v9', 'source_plan_path': str(source_path),
            'source_plan_sha256': source_sha, 'source_plan_fingerprint': source_plan['fingerprint'],
        }
        self.assertFalse(source_path.exists())
        self.assertFalse(manifest_path.exists())
        validator.validate_checkpoint(self.root, 2, self.b_result, self.c_result, context)
        lineage['origin']['source_plan_fingerprint'] = 'different'
        with self.assertRaisesRegex(ValueError, 'source plan binding changed'):
            validator.validate_checkpoint(self.root, 2, self.b_result, self.c_result, context)


class EmbeddedProvenanceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = {'kind': 'consolidated_plan', 'documents': {}}
        self.root_path = self.root / 'plan.json'
        self.source_path = self.root / 'plan.v9.json'
        self.manifest_path = self.root / 'deleted-configs/compatibility.json'
        self.root_plan = historical_plan({'historical': 7})
        self.source_plan = historical_plan({'historical': 9})
        root_sha = embedded_document(self.context, self.root_path, self.root_plan)
        source_sha = embedded_document(self.context, self.source_path, self.source_plan)
        self.source_binding = {'path': 'plan.v9.json', 'plan_sha256': source_sha,
                               'schema_version': 9, 'protocol': 'historical-v9',
                               'fingerprint': self.source_plan['fingerprint']}
        self.manifest = {'migration_id': 'historical-migration',
                         'target': {'fingerprint': self.source_plan['fingerprint']},
                         'origin_plans': {'v9': self.source_binding}}
        manifest_sha = embedded_document(self.context, self.manifest_path, self.manifest)
        self.reference = {
            'manifest_path': str(self.manifest_path), 'manifest_sha256': manifest_sha,
            'migration_id': self.manifest['migration_id'],
            'target_fingerprint': self.source_plan['fingerprint'],
            'legacy_plan_path': str(self.root_path), 'legacy_plan_sha256': root_sha,
            'legacy_fingerprint': self.root_plan['fingerprint'],
            'origin_plans': {'v9': dict(self.source_binding, resolved_path=str(self.source_path))},
        }
        # The original plan path now belongs to the consolidated plan, while
        # historical manifests and versioned plans no longer exist on disk.
        write_json(self.root_path, {'current': 'consolidated plan'})

    def validate(self):
        validator.validate_historical_reference(self.reference, self.context)

    def test_history_is_resolved_from_embedded_bytes_without_superseded_files(self):
        self.assertFalse(self.source_path.exists())
        self.assertFalse(self.manifest_path.exists())
        before = self.root_path.read_bytes()
        self.validate()
        self.assertEqual(before, self.root_path.read_bytes())

    def test_embedded_document_tampering_is_rejected(self):
        self.context['documents'][self.reference['manifest_sha256']]['json_text'] += ' '
        with self.assertRaisesRegex(ValueError, 'SHA|bytes|hash'):
            self.validate()

    def test_missing_snapshot_does_not_fall_back_to_disk(self):
        manifest = self.context['documents'].pop(self.reference['manifest_sha256'])
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text(manifest['json_text'], encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_reference_sha_path_and_origin_binding_are_rejected(self):
        cases = (
            lambda: self.reference.update(manifest_sha256='0' * 64),
            lambda: self.reference.update(manifest_path=str(self.root / 'unrecorded.json')),
            lambda: self.reference['origin_plans']['v9'].update(fingerprint='different'),
            lambda: self.reference.update(legacy_fingerprint='different'),
        )
        for change in cases:
            with self.subTest(change=change):
                original = deepcopy(self.reference)
                change()
                with self.assertRaises(ValueError):
                    self.validate()
                self.reference = original


class ConsolidatedLayoutValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename in ('plan.json', 'summary.csv'):
            (self.root / filename).touch()
        for group in ('A', 'B', 'C'):
            for trial in ('01', '02', '03'):
                (self.root / group / trial).mkdir(parents=True)

    def test_only_unified_plan_summary_and_three_group_directories_are_accepted(self):
        validator.validate_layout(self.root, 3)
        for filename in ('plan.v11.json', 'summary.v10.csv', 'invocations.jsonl'):
            with self.subTest(filename=filename):
                path = self.root / filename
                path.touch()
                with self.assertRaisesRegex(ValueError, 'must contain only'):
                    validator.validate_layout(self.root, 3)
                path.unlink()

    def test_extra_trial_and_substituted_symlink_are_rejected(self):
        extra = self.root / 'A/04'
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'trial directory set'):
            validator.validate_layout(self.root, 3)
        extra.rmdir()
        original = self.root / 'summary.csv'
        original.unlink()
        original.symlink_to(self.root / 'plan.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            validator.validate_layout(self.root, 3)


if __name__ == '__main__':
    unittest.main()
