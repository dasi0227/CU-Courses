"""Offline invariants for a single plan with content-bound embedded origins."""
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
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + '\n'


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class ConsolidatedCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'results'
        self.output.mkdir()
        config = runner.load_config(ROOT / 'configs/formula/smoke.json')
        config['repeats'] = 3
        data = runner.prepare_data(config, ROOT)
        self.plan = runner.build_plan(config, data, list('ABC'), mock=True)
        self.plan.pop('compatibility', None)
        self.origins, self.old_plans = {}, {}
        for schema in (7, 9, 10):
            old = deepcopy(self.plan)
            old.update(schema_version=schema, protocol=compatibility.HISTORICAL_PROTOCOLS[schema])
            old['source_sha256']['src/experiments/runner.py'] = text_sha(f'runner v{schema}')
            old['source_sha256']['src/experiments/compatibility.py'] = text_sha(f'compatibility v{schema}')
            if schema in (7, 9):
                old['config']['jev']['k'] = 5
                old['source_sha256']['src/experiments/jev.py'] = text_sha(f'JEV v{schema}')
            if schema == 7:
                old['config']['groups']['D'] = old['config']['groups'].pop('C')
                old['config']['groups']['C'] = {'kind': 'ace', 'max_num_rounds': 1}
            old['fingerprint'] = runner.digest({'config': old['config'], 'data': old['data'],
                                              'source': old['source_sha256'], 'mock': True})
            identifier = f'v{schema}'
            text = json_text(old)
            original_path = self.output / ('plan.json' if schema == 7 else f'plan.v{schema}.json')
            self.origins[identifier] = {
                **{field: old[field] for field in ('schema_version', 'protocol', 'fingerprint')},
                'plan_sha256': text_sha(text), 'original_path': str(original_path.resolve()),
                'json_text': text,
            }
            self.old_plans[identifier] = old
        self.recorded_manifests = {}
        v9_manifest = {
            'schema_version': 1, 'migration_id': 'fixture-v7-to-v9',
            'legacy': {field: self.origins['v7'][field]
                       for field in ('schema_version', 'protocol', 'fingerprint', 'plan_sha256')},
            'target': {field: deepcopy(self.old_plans['v9'][field]) for field in compatibility.PLAN_FIELDS},
        }
        self.v9_reference = self.record_manifest(v9_manifest, 'v9-manifest.json')
        self.preserved = {}
        for group, trial, source in (
            ('A', 1, 'v7'), ('A', 2, 'v9'), ('A', 3, 'v9'),
            ('B', 1, 'v7'), ('B', 2, 'v9'),
        ):
            result = self.base_result(group, trial, source)
            if source == 'v9':
                result['compatibility_source'] = deepcopy(self.v9_reference)
            if (group, trial) == ('B', 2):
                result['recovery'] = {'transport_timeout_override': 180}
            self.save_result(group, trial, source, result)
        bindings = {
            identifier: {field: binding[field]
                         for field in ('schema_version', 'protocol', 'fingerprint', 'plan_sha256')}
            for identifier, binding in self.origins.items() if identifier != 'v10'
        }
        for identifier, binding in bindings.items():
            binding['path'] = 'plan.json' if identifier == 'v7' else 'plan.v9.json'
        v10_manifest = {
            'schema_version': 2, 'migration_id': 'fixture-history-to-v10',
            'target': {field: deepcopy(self.old_plans['v10'][field]) for field in compatibility.PLAN_FIELDS},
            'origin_plans': bindings, 'preserved_results': deepcopy(self.preserved),
        }
        for entry in v10_manifest['preserved_results'].values():
            entry['original_status'] = entry.pop('snapshot_status')
        self.v10_reference = self.record_manifest(v10_manifest, 'v10-manifest.json')
        result = self.base_result('B', 3, 'v10')
        result['compatibility_source'] = deepcopy(self.v10_reference)
        self.save_result('B', 3, 'v10', result)
        for trial in (1, 2, 3):
            b_key = f'B/{trial:02d}'
            paired = read_json(self.output / b_key / 'result.json')
            entry = self.preserved[b_key]
            book = self.output / b_key / 'training/best_playbook.txt'
            origin = {'kind': 'current' if trial == 3 else 'legacy_compatible',
                      'fingerprint': paired['fingerprint'],
                      'result_sha256': entry['files_sha256']['result.json']}
            if trial != 3:
                binding = self.origins[entry['origin_id']]
                origin.update(
                    manifest_path=self.v10_reference['manifest_path'],
                    manifest_sha256=self.v10_reference['manifest_sha256'],
                    migration_id=v10_manifest['migration_id'], original_status='complete',
                    legacy_plan_sha256=self.origins['v7']['plan_sha256'],
                    source_plan_id=entry['origin_id'], source_plan_path=binding['original_path'],
                    source_plan_sha256=binding['plan_sha256'], source_plan_fingerprint=binding['fingerprint'])
            result = self.base_result('C', trial, 'v10')
            result.update(
                compatibility_source=deepcopy(self.v10_reference), playbook_source=str(book.resolve()),
                playbook_sha256=paired['playbook_sha256'], selection_k=10, selection_primitive='score',
                selection_ranking='score_desc_then_playbook_order',
                selection_confidence_used_for_ranking=False,
                b_checkpoint={'group': 'B', 'trial': trial, 'path': str(book.resolve()),
                              'sha256': paired['playbook_sha256'], 'original_fingerprint': paired['fingerprint'],
                              'origin': origin, 'shared_learning_llm_usage': self.learning_usage(),
                              'training_cost_included_in_C_incremental_usage': False,
                              'shared_learning_excludes_B_test': True})
            self.save_result('C', trial, 'v10', result)
        self.provenance = {'schema_version': 1, 'origin_plans': self.origins,
                           'recorded_manifests': self.recorded_manifests,
                           'source_changes': {}, 'preserved_results': self.preserved}
        self.rebind_source_changes()
        self.persist()
        self.original_bytes = {str(path.relative_to(self.output)): path.read_bytes()
                               for path in self.output.rglob('*') if path.is_file()}

    @staticmethod
    def learning_usage():
        return {'successful_calls': 1, 'attempts': 2, 'prompt_tokens': 30, 'completion_tokens': 10}

    def base_result(self, group, trial, source):
        return {'group': group, 'trial': trial, 'status': 'complete', 'mock': True,
                'formal_metrics': False, 'fingerprint': self.old_plans[source]['fingerprint'],
                'test': {'accuracy': 0.0, 'correct': 0, 'total': 2}, 'wall_seconds': 1.0,
                'llm': {'successful_calls': 1, 'attempts': 2, 'prompt_tokens': 30, 'completion_tokens': 10,
                        'stage_usage': {'training': self.learning_usage(),
                                        'test': {'successful_calls': 0, 'attempts': 0,
                                                 'prompt_tokens': 0, 'completion_tokens': 0}}},
                'llm_usage_including_shared_learning': {'successful_calls': 1, 'prompt_tokens': 30,
                                                       'completion_tokens': 10},
                'jev': {'successful_calls': 0}}

    def record_manifest(self, manifest, filename):
        text = json_text(manifest)
        sha = text_sha(text)
        path = str((self.output.parent / filename).resolve())
        self.recorded_manifests[sha] = {'original_path': path, 'json_text': text}
        reference = {'migration_id': manifest['migration_id'], 'manifest_path': path,
                     'manifest_sha256': sha, 'legacy_plan_path': self.origins['v7']['original_path'],
                     'legacy_plan_sha256': self.origins['v7']['plan_sha256'],
                     'legacy_fingerprint': self.origins['v7']['fingerprint'],
                     'target_fingerprint': manifest['target']['fingerprint']}
        if 'origin_plans' in manifest:
            reference['origin_plans'] = {
                key: {**binding, 'resolved_path': self.origins[key]['original_path']}
                for key, binding in manifest['origin_plans'].items()}
        return reference

    def save_result(self, group, trial, source, result):
        folder = self.output / group / f'{trial:02d}'
        folder.mkdir(parents=True)
        for name in compatibility.ARTIFACT_FILES[group] - {'result.json'}:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n', encoding='utf-8')
        if group == 'B':
            result['playbook_sha256'] = sha256(folder / 'training/best_playbook.txt')
            runner.write_json(folder / 'training/train_results.json', {
                'status': 'complete', 'best_checkpoint': {'playbook_sha256': result['playbook_sha256']}})
        runner.write_json(folder / 'result.json', result)
        self.preserved[f'{group}/{trial:02d}'] = {
            'origin_id': source, 'group': group, 'trial': trial, 'snapshot_status': 'complete',
            'original_fingerprint': result['fingerprint'],
            'files_sha256': {name: sha256(folder / name) for name in compatibility.ARTIFACT_FILES[group]}}

    def rebind_source_changes(self):
        for identifier, old in self.old_plans.items():
            before, after = old['source_sha256'], self.plan['source_sha256']
            self.provenance['source_changes'][identifier] = {
                name: {'old_sha256': before.get(name), 'new_sha256': after.get(name)}
                for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)}

    def persist(self):
        document = {**deepcopy(self.plan), 'provenance': deepcopy(self.provenance),
                    'provenance_sha256': runner.digest(self.provenance)}
        runner.write_json(self.output / 'plan.json', document)

    def load(self):
        return compatibility.load_compatibility(self.output, self.plan)

    def origin(self, group='B', trial=1, context=None):
        result = read_json(self.output / group / f'{trial:02d}' / 'result.json')
        return compatibility.validate_result_origin(self.output, group, trial, self.plan, result, context)

    def mutate_result(self, key, mutate):
        path = self.output / key / 'result.json'
        result = read_json(path)
        mutate(result)
        runner.write_json(path, result)
        self.preserved[key]['files_sha256']['result.json'] = sha256(path)
        self.persist()

    def test_unified_plan_preserves_all_real_fingerprints_and_bytes(self):
        context = self.load()
        self.assertEqual(context['kind'], 'consolidated_plan')
        for key, entry in self.preserved.items():
            group, trial = key.split('/')
            origin = self.origin(group, int(trial), context)
            self.assertEqual(origin['kind'], 'preserved')
            self.assertEqual(origin['fingerprint'], entry['original_fingerprint'])
            self.assertEqual(origin['source_plan_id'], entry['origin_id'])
        self.assertEqual(self.original_bytes, {str(path.relative_to(self.output)): path.read_bytes()
                                               for path in self.output.rglob('*') if path.is_file()})

    def test_old_paths_are_identities_and_never_read(self):
        self.assertFalse((self.output / 'plan.v9.json').exists())
        self.assertFalse((self.output / 'plan.v10.json').exists())
        context = self.load()
        old = compatibility.read_provenance_document(context, self.origins['v7']['original_path'],
                                                     self.origins['v7']['plan_sha256'])
        self.assertEqual(old['fingerprint'], self.old_plans['v7']['fingerprint'])
        self.assertNotEqual(old['fingerprint'], read_json(self.output / 'plan.json')['fingerprint'])
        # A stale or malicious document at the old manifest path has no effect.
        Path(self.v10_reference['manifest_path']).write_text('{"wrong":true}\n')
        self.assertEqual(context, self.load())

    def test_storage_names_are_always_the_same_two_files(self):
        self.assertEqual(compatibility.storage_names(self.output, self.plan, self.load()),
                         ('plan.json', 'summary.csv'))
        self.assertEqual(compatibility.storage_names(self.output, self.plan, None),
                         ('plan.json', 'summary.csv'))

    def test_fresh_native_roots_need_no_historical_provenance(self):
        fresh = self.output.parent / 'fresh'
        self.assertIsNone(compatibility.load_compatibility(fresh, self.plan))
        runner.write_json(fresh / 'plan.json', self.plan)
        self.assertIsNone(compatibility.load_compatibility(fresh, self.plan))

    def test_provenance_digest_binds_embedded_metadata(self):
        stored = read_json(self.output / 'plan.json')
        stored['provenance']['source_changes'].clear()
        runner.write_json(self.output / 'plan.json', stored)
        with self.assertRaisesRegex(ValueError, 'provenance digest'):
            self.load()

    def test_each_preserved_file_is_content_bound(self):
        for name, before in self.original_bytes.items():
            if name == 'plan.json':
                continue
            with self.subTest(file=name):
                path = self.output / name
                path.write_bytes(before + b'changed')
                with self.assertRaisesRegex(ValueError, 'preserved artifact changed'):
                    self.load()
                path.write_bytes(before)

    def test_artifact_set_must_be_exact(self):
        (self.output / 'A/01/extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'file set changed'):
            self.load()

    def test_all_nine_trials_must_be_listed(self):
        self.preserved.pop('A/03')
        self.persist()
        with self.assertRaisesRegex(ValueError, 'every final A/B/C trial'):
            self.load()

    def test_symlinked_artifacts_are_rejected(self):
        path = self.output / 'A/01/predictions.jsonl'
        target = self.output.parent / 'copied-predictions.jsonl'
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.load()

    def test_embedded_plan_hash_and_own_fingerprint_are_both_required(self):
        original = deepcopy(self.origins['v9'])
        self.origins['v9']['json_text'] += '\n'
        self.persist()
        with self.assertRaisesRegex(ValueError, 'document bytes changed'):
            self.load()
        self.origins['v9'] = original
        old = deepcopy(self.old_plans['v9'])
        old['config']['llm']['timeout'] = 181
        text = json_text(old)
        self.origins['v9'].update(json_text=text, plan_sha256=text_sha(text))
        self.persist()
        with self.assertRaisesRegex(ValueError, 'self-consistent'):
            self.load()

    def test_unknown_historical_protocol_cannot_be_rebound(self):
        old = deepcopy(self.old_plans['v9'])
        old.update(schema_version=8, protocol='formula-abc-compact-artifacts-bailian-v8')
        text = json_text(old)
        self.origins['v9'].update(schema_version=8, protocol=old['protocol'],
                                  json_text=text, plan_sha256=text_sha(text))
        self.persist()
        with self.assertRaisesRegex(ValueError, 'unrecognized historical plan'):
            self.load()

    def test_source_diff_requires_exact_review(self):
        self.provenance['source_changes']['v7']['src/experiments/jev.py']['new_sha256'] = '0' * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, 'source change review'):
            self.load()

    def test_prompt_changes_cannot_be_authorized_by_rebinding_provenance(self):
        self.plan['source_sha256']['src/core/generator.py'] = '0' * 64
        self.plan['fingerprint'] = runner.digest({'config': self.plan['config'], 'data': self.plan['data'],
                                                'source': self.plan['source_sha256'], 'mock': True})
        self.rebind_source_changes()
        self.persist()
        with self.assertRaisesRegex(ValueError, 'unreviewed source'):
            self.load()

    def test_current_config_data_mode_protocol_and_digest_are_bound(self):
        for mutate in (
            lambda plan: plan['config']['llm'].update(timeout=181),
            lambda plan: plan['data'].update(seed=999),
            lambda plan: plan['source_sha256'].update({'src/experiments/runner.py': '0' * 64}),
            lambda plan: plan.update(mode='live', formal_metrics=True),
            lambda plan: plan.update(fingerprint='0' * 64),
            lambda plan: plan.update(protocol='other-protocol'),
        ):
            with self.subTest(mutate=mutate):
                changed = deepcopy(self.plan)
                mutate(changed)
                with self.assertRaises(ValueError):
                    compatibility.load_compatibility(self.output, changed)

    def test_consistently_rebound_plan_cannot_change_b_training(self):
        self.plan['config']['groups']['B']['max_num_rounds'] = 1
        self.plan['fingerprint'] = runner.digest({'config': self.plan['config'], 'data': self.plan['data'],
                                                'source': self.plan['source_sha256'], 'mock': True})
        self.persist()
        with self.assertRaisesRegex(ValueError, 'only removing old C'):
            self.load()

    def test_result_fingerprint_and_metadata_cannot_be_rebound(self):
        for mutate, pattern in (
            (lambda result: result.update(fingerprint='0' * 64), 'historical fingerprint changed'),
            (lambda result: result.update(trial=3), 'group/trial'),
            (lambda result: result.update(status='failed'), 'status changed'),
            (lambda result: result.update(mock=False), 'execution mode'),
        ):
            with self.subTest(pattern=pattern):
                before = (self.output / 'A/01/result.json').read_bytes()
                self.mutate_result('A/01', mutate)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.load()
                (self.output / 'A/01/result.json').write_bytes(before)
                self.preserved['A/01']['files_sha256']['result.json'] = text_sha(before.decode())
                self.persist()

    def test_each_trial_must_name_its_actual_source_plan(self):
        self.preserved['B/02']['origin_id'] = 'v7'
        self.persist()
        with self.assertRaisesRegex(ValueError, 'fingerprint must match'):
            self.load()

    def test_c_lineage_score_policy_and_learning_cost_cannot_be_rebound(self):
        for mutate, pattern in (
            (lambda result: result['b_checkpoint'].update(trial=2), 'same-number B'),
            (lambda result: result['b_checkpoint'].update(sha256='0' * 64), 'playbook source or SHA'),
            (lambda result: result['b_checkpoint'].update(original_fingerprint='0' * 64), 'original fingerprint'),
            (lambda result: result['b_checkpoint']['origin'].update(result_sha256='0' * 64), 'result origin'),
            (lambda result: result['b_checkpoint']['origin'].update(source_plan_sha256='0' * 64), 'source plan changed'),
            (lambda result: result.update(selection_k=5), 'Score top-10 selection'),
            (lambda result: result.update(selection_confidence_used_for_ranking=True), 'Score top-10 selection'),
            (lambda result: result['b_checkpoint']['shared_learning_llm_usage'].update(attempts=999), 'training usage'),
        ):
            with self.subTest(pattern=pattern):
                before = (self.output / 'C/01/result.json').read_bytes()
                self.mutate_result('C/01', mutate)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.load()
                (self.output / 'C/01/result.json').write_bytes(before)
                self.preserved['C/01']['files_sha256']['result.json'] = text_sha(before.decode())
                self.persist()

    def test_native_c_checkpoint_still_requires_same_actual_source_fingerprint(self):
        self.mutate_result('C/01', lambda result: result['b_checkpoint']['origin'].update(kind='current'))
        with self.assertRaisesRegex(ValueError, 'share its actual v10 fingerprint'):
            self.load()

    def test_old_reference_manifest_path_sha_and_target_are_validated(self):
        self.mutate_result('B/03', lambda result: result['compatibility_source'].update(manifest_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'not explicitly recorded'):
            self.load()

    def test_embedded_manifest_bytes_are_checked_even_with_provenance_digest_updated(self):
        document = self.recorded_manifests[self.v10_reference['manifest_sha256']]
        document['json_text'] += '\n'
        self.persist()
        with self.assertRaisesRegex(ValueError, 'document bytes changed'):
            self.load()

    def test_document_parser_checks_path_sha_json_object_and_bytes(self):
        context = self.load()
        path = self.origins['v7']['original_path']
        sha = self.origins['v7']['plan_sha256']
        with self.assertRaisesRegex(ValueError, 'not explicitly recorded'):
            compatibility.read_provenance_document(context, path + '.other', sha)
        for raw in ('not json', '[]'):
            digest = text_sha(raw)
            bad = {'kind': 'consolidated_plan', 'documents': {
                digest: {'original_path': path, 'json_text': raw}}}
            with self.assertRaises(ValueError):
                compatibility.read_provenance_document(bad, path, digest)

    def test_context_is_revalidated_between_reads(self):
        context = self.load()
        self.provenance['source_changes']['v9'].clear()
        self.persist()
        with self.assertRaisesRegex(ValueError, 'source change review'):
            self.origin(context=context)

    def test_b_checkpoint_reuses_its_exact_paired_source_and_leaves_artifacts_unchanged(self):
        for trial, identifier in ((1, 'v7'), (2, 'v9'), (3, 'v10')):
            book, lineage = runner.b_checkpoint(self.output, trial, self.plan)
            self.assertEqual(book, '{}\n')
            self.assertEqual(lineage['origin']['kind'], 'preserved')
            self.assertEqual(lineage['origin']['source_plan_id'], identifier)
            self.assertEqual(lineage['shared_learning_llm_usage'], self.learning_usage())
        self.assertEqual(self.original_bytes, {str(path.relative_to(self.output)): path.read_bytes()
                                               for path in self.output.rglob('*') if path.is_file()})


if __name__ == '__main__':
    unittest.main()
