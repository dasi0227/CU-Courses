"""Validate final ABC snapshots using provenance embedded in one plan.json.

Original plan and manifest texts remain content bound inside the unified plan.
Their paths identify historical documents; validation never opens those paths.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

TARGET_SCHEMA_VERSION = 11
TARGET_PROTOCOL = 'formula-abc-score-top10-final-results-bailian-v11'
HISTORICAL_PROTOCOLS = {
    7: 'formula-abcd-compact-artifacts-bailian-v7',
    9: 'formula-abc-score-compact-artifacts-bailian-v9',
    10: 'formula-abc-score-top10-compact-artifacts-bailian-v10',
}
SOURCE_CHANGE_ALLOWLIST = frozenset({
    'src/experiments/data.py', 'src/experiments/runner.py',
    'src/experiments/runtime.py', 'src/experiments/compatibility.py',
    'eval/finance/experiments.py', 'src/experiments/jev.py',
})
ARTIFACT_FILES = {
    'A': frozenset({'result.json', 'llm_calls.jsonl', 'predictions.jsonl'}),
    'B': frozenset({'result.json', 'llm_calls.jsonl', 'predictions.jsonl',
                    'training/train_results.json', 'training/best_playbook.txt'}),
    'C': frozenset({'result.json', 'llm_calls.jsonl', 'predictions.jsonl', 'jev_audit.jsonl'}),
}
PLAN_FIELDS = ('schema_version', 'protocol', 'fingerprint', 'config', 'data',
               'source_sha256', 'mode', 'formal_metrics')
_SHA = re.compile(r'^[0-9a-f]{64}$')


def _require(condition, message):
    if not condition:
        raise ValueError('Compatibility validation failed: ' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path):
    _require(not path.is_symlink() and path.is_file(), f'plain JSON file required: {path}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Compatibility validation failed: cannot read {path}: {exc}') from exc
    _require(isinstance(value, dict), f'JSON object required: {path}')
    return value


def _valid_sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _plain_relative(name):
    return (isinstance(name, str) and bool(name) and '\\' not in name
            and not Path(name).is_absolute() and all(part not in ('', '.', '..') for part in name.split('/')))


def _plain_child(root, name):
    _require(_plain_relative(name), f'unsafe relative path: {name!r}')
    path = root
    for part in name.split('/'):
        path = path / part
        _require(not path.is_symlink(), f'symlink is not a preserved artifact: {path}')
    return path


def _check_plan_digest(plan):
    _require(plan.get('mode') in ('live', 'mock'), 'plan mode must be live or mock')
    for field in ('config', 'data', 'source_sha256'):
        _require(isinstance(plan.get(field), dict), f'plan {field} must be an object')
    source = plan['source_sha256']
    _require(all(_plain_relative(name) and _valid_sha(sha) for name, sha in source.items()),
             'source hashes must be relative paths with SHA256 values')
    expected = _digest({'config': plan['config'], 'data': plan['data'],
                        'source': source, 'mock': plan['mode'] == 'mock'})
    _require(plan.get('fingerprint') == expected, 'plan fingerprint is not self-consistent')


def _check_current_plan(existing, plan):
    _check_plan_digest(existing)
    _check_plan_digest(plan)
    _require(all(existing.get(field) == plan.get(field) for field in PLAN_FIELDS),
             'existing current plan does not match configuration, data, source, mode and protocol')


def _check_result_metadata(result, group, trial, plan, accepted_statuses=('complete',)):
    _require(group in ('A', 'B', 'C') and isinstance(trial, int) and not isinstance(trial, bool),
             'invalid current group or trial')
    _require(1 <= trial <= plan['config']['repeats'], 'trial is outside the plan')
    _require(result.get('group') == group and type(result.get('trial')) is int
             and result['trial'] == trial, 'result group/trial does not match its folder')
    _require(result.get('status') in accepted_statuses, 'only completed results are accepted'
             if accepted_statuses == ('complete',) else 'historical result status changed')
    _require(type(result.get('mock')) is bool and result['mock'] == (plan['mode'] == 'mock'),
             'result execution mode does not match the plan')
    _require(type(result.get('formal_metrics')) is bool
             and result['formal_metrics'] == plan['formal_metrics'],
             'result formal-metrics provenance does not match the plan')


def read_provenance_document(context, path, expected_sha):
    """Read a hash-bound historical document from memory, never its old path."""
    _require(isinstance(context, dict) and context.get('kind') == 'consolidated_plan',
             'consolidated provenance context is required')
    _require(isinstance(path, (str, Path)) and Path(path).is_absolute()
             and _valid_sha(expected_sha), 'historical document requires an absolute identity and SHA256')
    documents = context.get('documents')
    _require(isinstance(documents, dict), 'embedded historical documents are required')
    document = documents.get(expected_sha)
    _require(isinstance(document, dict) and document.get('original_path') == str(path),
             'historical document path and SHA are not explicitly recorded')
    raw = document.get('json_text')
    _require(isinstance(raw, str) and hashlib.sha256(raw.encode('utf-8')).hexdigest() == expected_sha,
             'embedded historical document bytes changed')
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('Compatibility validation failed: embedded historical document is invalid JSON') from exc
    _require(isinstance(value, dict), 'embedded historical document must be a JSON object')
    return value


def _check_recorded_reference(root, context, reference, source_plan):
    """Verify an immutable old reference using embedded historical documents."""
    _require(isinstance(reference, dict), 'recorded compatibility source must be an object')
    manifest = read_provenance_document(context, reference.get('manifest_path'),
                                        reference.get('manifest_sha256'))
    _require(manifest.get('migration_id') == reference.get('migration_id'),
             'recorded compatibility migration changed')
    target = manifest.get('target')
    _require(isinstance(target, dict)
             and all(target.get(field) == source_plan.get(field) for field in PLAN_FIELDS)
             and reference.get('target_fingerprint') == source_plan['fingerprint'],
             'recorded compatibility target does not match the actual source plan')
    legacy_path = reference.get('legacy_plan_path')
    legacy = read_provenance_document(context, legacy_path, reference.get('legacy_plan_sha256'))
    _check_plan_digest(legacy)
    _require(legacy_path == str((root / 'plan.json').resolve())
             and reference.get('legacy_fingerprint') == legacy['fingerprint'],
             'recorded compatibility root plan changed')
    if 'origin_plans' in reference:
        origins = manifest.get('origin_plans')
        _require(isinstance(origins, dict) and isinstance(reference['origin_plans'], dict)
                 and set(reference['origin_plans']) == set(origins),
                 'recorded compatibility origin plans changed')
        for origin_id, binding in origins.items():
            _require(isinstance(binding, dict) and _plain_relative(binding.get('path')),
                     'recorded compatibility source plan binding is invalid')
            recorded = reference['origin_plans'][origin_id]
            source_path = str((root / binding['path']).resolve())
            old = read_provenance_document(context, source_path, binding.get('plan_sha256'))
            _check_plan_digest(old)
            _require(isinstance(recorded, dict)
                     and recorded == {**binding, 'resolved_path': source_path}
                     and all(binding.get(field) == old.get(field)
                             for field in ('schema_version', 'protocol', 'fingerprint')),
                     'recorded compatibility source plan identity changed')
    else:
        binding = manifest.get('legacy')
        _require(isinstance(binding, dict)
                 and binding.get('plan_sha256') == reference['legacy_plan_sha256']
                 and binding.get('fingerprint') == legacy['fingerprint'],
                 'recorded compatibility legacy binding changed')
    return manifest


def _check_entry(root, key, entry, old_plan, target_plan, context):
    _require(isinstance(entry, dict), f'invalid preserved result entry: {key}')
    group, trial = entry.get('group'), entry.get('trial')
    _require(isinstance(group, str) and group in ARTIFACT_FILES and type(trial) is int and trial >= 1,
             'only explicitly listed historical A/B/C trials may be preserved')
    _require(group != 'C' or old_plan.get('schema_version') == 10,
             'only native v10 Score top-10 C trials may be preserved')
    _require(key == f'{group}/{trial:02d}', 'preserved result path must match group/trial')
    status_field = 'snapshot_status'
    _require(entry.get(status_field) == 'complete',
             'preserved final snapshot must be complete')
    _require(entry.get('original_fingerprint') == old_plan['fingerprint'],
             'preserved result fingerprint must match the historical plan')
    files = entry.get('files_sha256')
    _require(isinstance(files, dict) and set(files) == ARTIFACT_FILES[group]
             and all(_valid_sha(sha) for sha in files.values()),
             f'preserved {key} must list the complete known artifact set')
    folder = _plain_child(root, key)
    _require(folder.is_dir(), f'preserved result folder is missing: {key}')
    actual = set()
    for path in folder.rglob('*'):
        _require(not path.is_symlink(), f'symlink is not a preserved artifact: {path}')
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    _require(actual == set(files), f'preserved result file set changed: {key}')
    for name, sha in files.items():
        path = _plain_child(folder, name)
        _require(path.is_file() and _file_sha(path) == sha, f'preserved artifact changed: {key}/{name}')
    result = _read_json(folder / 'result.json')
    _require(result.get('status') == entry[status_field], 'historical result status changed')
    _check_result_metadata(result, group, trial, target_plan, (entry[status_field],))
    _require(result.get('fingerprint') == old_plan['fingerprint'], f'historical fingerprint changed: {key}')
    if result.get('compatibility_source') is not None:
        _check_recorded_reference(root, context, result['compatibility_source'], old_plan)
    if group == 'B':
        book_sha = files['training/best_playbook.txt']
        _require(result.get('playbook_sha256') == book_sha, 'historical B playbook hash does not match result')
        training = _read_json(folder / 'training/train_results.json')
        best = training.get('best_checkpoint')
        if isinstance(best, dict) and 'playbook_sha256' in best:
            _require(best['playbook_sha256'] == book_sha, 'historical B best checkpoint hash does not match')
    if group == 'C':
        _require(result.get('selection_k') == 10 and result.get('selection_primitive') == 'score'
                 and result.get('selection_ranking') == 'score_desc_then_playbook_order'
                 and result.get('selection_confidence_used_for_ranking') is False,
                 'historical C must retain the Score top-10 selection policy')
    return result


def _check_c_lineage(root, key, result, preserved, checked_results, checked_plans, context):
    trial = result['trial']
    b_key = f'B/{trial:02d}'
    _require(b_key in preserved and b_key in checked_results,
             f'preserved {key} requires its explicitly preserved paired B trial')
    b_entry, b_result = preserved[b_key], checked_results[b_key]
    _require(b_result.get('status') == 'complete', 'historical C requires a completed paired B')
    checkpoint = result.get('b_checkpoint')
    _require(isinstance(checkpoint, dict) and checkpoint.get('group') == 'B'
             and type(checkpoint.get('trial')) is int and checkpoint['trial'] == trial,
             'historical C must retain its same-number B checkpoint')
    book_path = _plain_child(root, f'{b_key}/training/best_playbook.txt')
    book_sha = b_entry['files_sha256']['training/best_playbook.txt']
    _require(checkpoint.get('path') == result.get('playbook_source') == str(book_path.resolve())
             and checkpoint.get('sha256') == result.get('playbook_sha256') == book_sha
             and b_result.get('playbook_sha256') == book_sha,
             'historical C paired B playbook source or SHA changed')
    b_plan = checked_plans[b_entry['origin_id']]
    _require(checkpoint.get('original_fingerprint') == b_result['fingerprint'] == b_plan['fingerprint'],
             'historical C paired B original fingerprint changed')
    origin = checkpoint.get('origin')
    _require(isinstance(origin, dict) and origin.get('fingerprint') == b_result['fingerprint']
             and origin.get('result_sha256') == b_entry['files_sha256']['result.json'],
             'historical C paired B result origin changed')
    if origin.get('kind') == 'current':
        _require(b_result['fingerprint'] == result['fingerprint'],
                 'historical C native B checkpoint must share its actual v10 fingerprint')
    else:
        _require(origin.get('kind') == 'legacy_compatible', 'unrecognized historical C checkpoint origin')
        c_plan = checked_plans[preserved[key]['origin_id']]
        _require(isinstance(result.get('compatibility_source'), dict),
                 'historical C legacy checkpoint requires its recorded compatibility source')
        old_manifest = _check_recorded_reference(root, context, result['compatibility_source'], c_plan)
        _require(origin.get('manifest_path') == result['compatibility_source']['manifest_path']
                 and origin.get('manifest_sha256') == result['compatibility_source']['manifest_sha256']
                 and origin.get('migration_id') == old_manifest.get('migration_id')
                 and origin.get('legacy_plan_sha256') == result['compatibility_source']['legacy_plan_sha256']
                 and origin.get('original_status') == 'complete',
                 'historical C checkpoint compatibility reference changed')
        source_id = origin.get('source_plan_id')
        old_binding = old_manifest.get('origin_plans', {}).get(source_id)
        _require(isinstance(old_binding, dict), 'historical C checkpoint source plan is unlisted')
        _require(_plain_relative(old_binding.get('path')), 'historical C source plan path is invalid')
        source_path = str((root / old_binding['path']).resolve())
        source = read_provenance_document(context, source_path, old_binding.get('plan_sha256'))
        _require(origin.get('source_plan_path') == source_path
                 and origin.get('source_plan_sha256') == old_binding.get('plan_sha256')
                 and origin.get('source_plan_fingerprint') == old_binding.get('fingerprint')
                     == b_plan['fingerprint'] == source.get('fingerprint'),
                 'historical C checkpoint source plan changed')
        old_entry = old_manifest.get('preserved_results', {}).get(b_key)
        _require(isinstance(old_entry, dict) and old_entry.get('files_sha256') == b_entry['files_sha256']
                 and old_entry.get('original_fingerprint') == b_result['fingerprint'],
                 'historical C original B training artifacts changed')
    stages = [usage for stage, usage in b_result['llm']['stage_usage'].items() if stage != 'test']
    expected_usage = {field: sum(stage[field] for stage in stages)
                      for field in ('successful_calls', 'attempts', 'prompt_tokens', 'completion_tokens')}
    _require(checkpoint.get('shared_learning_llm_usage') == expected_usage
             and checkpoint.get('training_cost_included_in_C_incremental_usage') is False
             and checkpoint.get('shared_learning_excludes_B_test') is True,
             'historical C paired B training usage changed')


def _converted_config(old_plan):
    """Apply only the reviewed arm migration and fixed top-5 to top-10 change."""
    converted = copy.deepcopy(old_plan['config'])
    groups = converted.get('groups')
    schema = old_plan.get('schema_version')
    if schema == 10:
        _require(isinstance(groups, dict) and set(groups) == set('ABC')
                 and converted.get('jev', {}).get('k') == 10,
                 'historical v10 must retain its ABC Score top-10 configuration')
        return converted
    if schema == 7:
        _require(isinstance(groups, dict) and set(groups) == set('ABCD'),
                 'historical v7 groups must be ABCD')
        _require(groups['C'].get('kind') == 'ace' and groups['C'].get('max_num_rounds') == 1,
                 'historical C must be the removed one-round arm')
        _require(groups['D'].get('kind') == 'jev_filter' and groups['D'].get('depends_on') == 'B',
                 'historical D must be the B-dependent JEV arm')
        converted['groups'] = {'A': groups['A'], 'B': groups['B'], 'C': groups['D']}
    else:
        _require(isinstance(groups, dict) and set(groups) == set('ABC'),
                 'historical v9 groups must be ABC')
    jev = converted.get('jev')
    _require(isinstance(jev, dict) and type(jev.get('k')) is int and jev['k'] == 5,
             'historical JEV selection must be fixed top-5')
    jev['k'] = 10
    return converted


def load_compatibility(output_root, plan):
    """Validate a native or consolidated plan and every preserved final file."""
    root = Path(output_root)
    plan_path = root / 'plan.json'
    _require(not plan_path.is_symlink(), 'root plan must not be a symlink')
    _check_plan_digest(plan)
    _require(plan.get('schema_version') == TARGET_SCHEMA_VERSION
             and plan.get('protocol') == TARGET_PROTOCOL,
             'current target must use the ABC final-results v11 protocol')
    _require(type(plan.get('formal_metrics')) is bool, 'plan formal_metrics must be a boolean')
    if not plan_path.exists():
        return None
    existing = _read_json(plan_path)
    _check_current_plan(existing, plan)
    provenance = existing.get('provenance')
    if provenance is None:
        _require('provenance_sha256' not in existing, 'provenance hash requires provenance')
        return None
    _require(isinstance(provenance, dict) and provenance.get('schema_version') == 1,
             'unrecognized consolidated provenance schema')
    _require(existing.get('provenance_sha256') == _digest(provenance),
             'consolidated provenance digest changed')
    origins = provenance.get('origin_plans')
    _require(isinstance(origins, dict) and bool(origins), 'explicit historical origin plans are required')
    manifests = provenance.get('recorded_manifests')
    _require(isinstance(manifests, dict), 'recorded manifests must be embedded objects')
    documents = {}
    for sha, document in manifests.items():
        _require(_valid_sha(sha) and isinstance(document, dict), 'invalid recorded manifest document')
        documents[sha] = copy.deepcopy(document)
    for binding in origins.values():
        _require(isinstance(binding, dict), 'invalid historical origin binding')
        sha = binding.get('plan_sha256')
        _require(_valid_sha(sha), 'historical plan requires its exact original SHA256')
        document = {field: binding.get(field) for field in ('original_path', 'json_text')}
        _require(sha not in documents or documents[sha] == document,
                 'one historical document SHA has conflicting identities')
        documents[sha] = document
    context = {
        'kind': 'consolidated_plan', 'target_fingerprint': plan['fingerprint'],
        'provenance_sha256': existing['provenance_sha256'],
        'origin_plans': copy.deepcopy(origins), 'documents': documents,
    }
    for sha, document in documents.items():
        read_provenance_document(context, document.get('original_path'), sha)
    source_changes = provenance.get('source_changes')
    _require(isinstance(source_changes, dict) and set(source_changes) == set(origins),
             'source change review must bind every origin plan exactly')
    checked_plans, seen_identities = {}, set()
    for origin_id, binding in origins.items():
        _require(isinstance(origin_id, str) and bool(origin_id.strip()), 'invalid historical origin identifier')
        identity = (binding.get('original_path'), binding['plan_sha256'])
        _require(identity not in seen_identities, 'duplicate historical plan identity')
        seen_identities.add(identity)
        old = read_provenance_document(context, *identity)
        _check_plan_digest(old)
        schema = old.get('schema_version')
        _require(type(schema) is int and schema in HISTORICAL_PROTOCOLS
                 and old.get('protocol') == HISTORICAL_PROTOCOLS[schema],
                 'unrecognized historical plan protocol')
        _require(all(binding.get(field) == old.get(field)
                     for field in ('schema_version', 'protocol', 'fingerprint')),
                 'historical plan identity does not match its embedded document')
        _require(_converted_config(old) == plan['config'],
                 'only removing old C/renaming D, fixed top-5 to top-10, and final result snapshots are compatible')
        _require(old['data'] == plan['data'] and old['mode'] == plan['mode']
                 and type(old.get('formal_metrics')) is bool
                 and old['formal_metrics'] == plan['formal_metrics'],
                 'historical and target data/execution provenance must match')
        before, after = old['source_sha256'], plan['source_sha256']
        changes = {name: {'old_sha256': before.get(name), 'new_sha256': after.get(name)}
                   for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)}
        _require(set(changes) <= SOURCE_CHANGE_ALLOWLIST, 'unreviewed source file changed')
        _require(source_changes[origin_id] == changes,
                 'source change review does not match the exact source diff')
        checked_plans[origin_id] = old
    preserved = provenance.get('preserved_results')
    expected = {f'{group}/{trial:02d}' for group in ARTIFACT_FILES
                for trial in range(1, plan['config']['repeats'] + 1)}
    _require(isinstance(preserved, dict) and set(preserved) == expected,
             'consolidated provenance must bind every final A/B/C trial exactly')
    context['preserved_results'] = copy.deepcopy(preserved)
    checked_results, used_origins = {}, set()
    for key, entry in preserved.items():
        _require(isinstance(entry, dict) and entry.get('origin_id') in checked_plans,
                 'preserved result must name an explicitly reviewed origin plan')
        origin_id = entry['origin_id']
        checked_results[key] = _check_entry(root, key, entry, checked_plans[origin_id], plan, context)
        used_origins.add(origin_id)
    _require(used_origins == set(checked_plans), 'every origin plan must bind preserved results')
    for key, result in checked_results.items():
        if result['group'] == 'C':
            _check_c_lineage(root, key, result, preserved, checked_results, checked_plans, context)
    return context


def validate_result_origin(output_root, group, trial, plan, result, compatibility=None):
    """Verify current or preserved origins while retaining real fingerprints."""
    root = Path(output_root)
    context = load_compatibility(root, plan)
    if compatibility is not None:
        _require(context == compatibility, 'compatibility context changed after validation')
    _check_result_metadata(result, group, trial, plan)
    path = _plain_child(root, f'{group}/{trial:02d}/result.json')
    stored = _read_json(path)
    _require(stored == result, 'supplied result differs from the persisted result')
    fingerprint = result.get('fingerprint')
    if fingerprint == plan['fingerprint']:
        return {'kind': 'current', 'fingerprint': fingerprint, 'result_sha256': _file_sha(path)}
    _require(context is not None and group in ARTIFACT_FILES,
             'result fingerprint is not current or explicitly preserved')
    key = f'{group}/{trial:02d}'
    entry = context['preserved_results'].get(key)
    _require(entry is not None, f'historical result is not explicitly preserved: {key}')
    binding = context['origin_plans'][entry['origin_id']]
    _require(fingerprint == binding['fingerprint'] == entry['original_fingerprint'],
             'historical result fingerprint does not match its preserved origin')
    return {'kind': 'preserved', 'fingerprint': fingerprint, 'result_sha256': _file_sha(path),
            'source_plan_id': entry['origin_id'], 'source_plan_sha256': binding['plan_sha256'],
            'source_plan_fingerprint': binding['fingerprint'],
            'provenance_sha256': context['provenance_sha256']}


def storage_names(output_root, plan, compatibility):
    """All runs use the same two root management files."""
    if compatibility is not None:
        _require(compatibility.get('target_fingerprint') == plan.get('fingerprint'),
                 'storage context target does not match the plan')
    return 'plan.json', 'summary.csv'
