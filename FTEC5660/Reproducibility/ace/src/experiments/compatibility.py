"""Immutable, reviewed provenance from historical plans to ABC Score top-10.

Compatibility binds each preserved result to its actual source plan and bytes.
Failed historical trials are validated and preserved but cannot provide a
completed result or B checkpoint.  All functions are local and read only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

from .data import PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / 'configs/formula/compatibility/history-to-abc-score-top10-v10.json'
TARGET_SCHEMA_VERSION = 10
TARGET_PROTOCOL = 'formula-abc-score-top10-compact-artifacts-bailian-v10'
HISTORICAL_PROTOCOLS = {
    7: 'formula-abcd-compact-artifacts-bailian-v7',
    9: 'formula-abc-score-compact-artifacts-bailian-v9',
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


def _check_entry(root, key, entry, old_plan, target_plan):
    _require(isinstance(entry, dict), f'invalid preserved result entry: {key}')
    group, trial = entry.get('group'), entry.get('trial')
    _require(isinstance(group, str) and group in ARTIFACT_FILES and type(trial) is int and trial >= 1,
             'only explicitly listed historical A/B trials may be preserved')
    _require(key == f'{group}/{trial:02d}', 'preserved result path must match group/trial')
    _require(entry.get('original_status') in ('complete', 'failed'),
             'historical status must explicitly preserve complete or failed')
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
    _require(result.get('status') == entry['original_status'], 'historical result status changed')
    _check_result_metadata(result, group, trial, target_plan, (entry['original_status'],))
    _require(result.get('fingerprint') == old_plan['fingerprint'], f'historical fingerprint changed: {key}')
    if group == 'B':
        book_sha = files['training/best_playbook.txt']
        _require(result.get('playbook_sha256') == book_sha, 'historical B playbook hash does not match result')
        training = _read_json(folder / 'training/train_results.json')
        best = training.get('best_checkpoint')
        if isinstance(best, dict) and 'playbook_sha256' in best:
            _require(best['playbook_sha256'] == book_sha, 'historical B best checkpoint hash does not match')
    return result


def _converted_config(old_plan):
    """Apply only the reviewed arm migration and fixed top-5 to top-10 change."""
    converted = copy.deepcopy(old_plan['config'])
    groups = converted.get('groups')
    schema = old_plan.get('schema_version')
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


def load_compatibility(output_root, plan, manifest_path=None):
    """Validate all exact source plans and preserved artifacts without writes.

    Return provenance for a historical root, or ``None`` for an empty root or
    a matching native root.  Historic fingerprints and recovery metadata are
    immutable inputs, never replaced by the target fingerprint.
    """
    root = Path(output_root)
    plan_path = root / 'plan.json'
    _require(not plan_path.is_symlink(), 'root plan must not be a symlink')
    _check_plan_digest(plan)
    _require(plan.get('schema_version') == TARGET_SCHEMA_VERSION
             and plan.get('protocol') == TARGET_PROTOCOL,
             'current target must be the reviewed ABC Score top-10 v10 protocol')
    _require(type(plan.get('formal_metrics')) is bool, 'plan formal_metrics must be a boolean')
    if not plan_path.exists():
        return None
    existing = _read_json(plan_path)
    if existing.get('schema_version') == plan.get('schema_version'):
        _check_current_plan(existing, plan)
        return None
    _require(type(existing.get('schema_version')) is int
             and existing.get('protocol') == HISTORICAL_PROTOCOLS.get(existing.get('schema_version')),
             'unrecognized historical plan protocol')
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    manifest = _read_json(path)
    _require(manifest.get('schema_version') == 2, 'unrecognized compatibility manifest schema')
    _require(isinstance(manifest.get('migration_id'), str) and bool(manifest['migration_id'].strip()),
             'compatibility migration_id must be a nonempty string')
    target = manifest.get('target')
    _require(isinstance(target, dict) and all(target.get(field) == plan.get(field) for field in PLAN_FIELDS),
             'target configuration, data, source, mode or protocol changed after review')
    origins = manifest.get('origin_plans')
    _require(isinstance(origins, dict) and bool(origins), 'explicit historical origin plans are required')
    source_changes = manifest.get('source_changes')
    _require(isinstance(source_changes, dict) and set(source_changes) == set(origins),
             'source change review must bind every origin plan exactly')
    checked_plans, context_origins = {}, {}
    seen_paths = set()
    root_origin = None
    for origin_id, binding in origins.items():
        _require(isinstance(origin_id, str) and bool(origin_id.strip()) and isinstance(binding, dict),
                 'invalid historical origin identifier or binding')
        origin_path = _plain_child(root, binding.get('path'))
        resolved = str(origin_path.resolve())
        _require(resolved not in seen_paths, 'duplicate historical origin plan path')
        seen_paths.add(resolved)
        _require(_valid_sha(binding.get('plan_sha256')) and origin_path.is_file()
                 and _file_sha(origin_path) == binding['plan_sha256'],
                 'historical plan bytes changed')
        old = _read_json(origin_path)
        _check_plan_digest(old)
        schema = old.get('schema_version')
        _require(type(schema) is int and schema in HISTORICAL_PROTOCOLS
                 and old.get('protocol') == HISTORICAL_PROTOCOLS[schema],
                 'unrecognized historical plan protocol')
        _require(all(binding.get(field) == old.get(field)
                     for field in ('schema_version', 'protocol', 'fingerprint')),
                 'historical plan identity does not match the manifest')
        _require(_converted_config(old) == plan['config'],
                 'only removing old C/renaming D and fixed top-5 to top-10 is compatible')
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
        context_origins[origin_id] = {**copy.deepcopy(binding), 'resolved_path': resolved}
        if origin_path == plan_path:
            root_origin = origin_id
    _require(root_origin is not None, 'root historical plan must be explicitly bound')
    preserved = manifest.get('preserved_results')
    _require(isinstance(preserved, dict) and bool(preserved), 'explicit preserved artifact entries are required')
    used_origins = set()
    for key, entry in preserved.items():
        _require(isinstance(entry, dict) and isinstance(entry.get('origin_id'), str)
                 and entry['origin_id'] in checked_plans,
                 'preserved result must name an explicitly reviewed origin plan')
        origin_id = entry['origin_id']
        _check_entry(root, key, entry, checked_plans[origin_id], plan)
        used_origins.add(origin_id)
    _require(used_origins == set(checked_plans), 'every origin plan must bind preserved results')
    return {
        'migration_id': manifest['migration_id'],
        'manifest_path': str(path.resolve()), 'manifest_sha256': _file_sha(path),
        'legacy_plan_path': str(plan_path.resolve()), 'legacy_plan_sha256': _file_sha(plan_path),
        'legacy_fingerprint': existing['fingerprint'], 'target_fingerprint': plan['fingerprint'],
        'origin_plans': context_origins, 'root_origin_id': root_origin,
        'preserved_results': copy.deepcopy(preserved),
    }


def validate_result_origin(output_root, group, trial, plan, result, compatibility=None):
    """Verify native or explicitly preserved provenance without modifying it."""
    root = Path(output_root)
    # Re-read the bindings instead of trusting a context retained across reads.
    context = load_compatibility(root, plan,
                                 compatibility.get('manifest_path') if compatibility is not None else None)
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
             'result fingerprint is not current or explicitly compatible')
    key = f'{group}/{trial:02d}'
    entry = context['preserved_results'].get(key)
    _require(entry is not None, f'historical result is not explicitly preserved: {key}')
    binding = context['origin_plans'][entry['origin_id']]
    _require(fingerprint == binding['fingerprint'] == entry['original_fingerprint'],
             'historical result fingerprint does not match its preserved origin')
    return {'kind': 'legacy_compatible', 'fingerprint': fingerprint,
            'result_sha256': _file_sha(path), 'manifest_sha256': context['manifest_sha256'],
            'legacy_plan_sha256': context['legacy_plan_sha256'],
            'migration_id': context['migration_id'], 'manifest_path': context['manifest_path'],
            'source_plan_id': entry['origin_id'], 'source_plan_path': binding['resolved_path'],
            'source_plan_sha256': binding['plan_sha256'],
            'source_plan_fingerprint': binding['fingerprint'],
            'original_status': entry['original_status']}


def storage_names(output_root, plan, compatibility):
    """Use distinct root bookkeeping files when preserving a legacy plan."""
    if compatibility is not None:
        _require(compatibility.get('target_fingerprint') == plan.get('fingerprint'),
                 'storage context target does not match the plan')
        return 'plan.v10.json', 'invocations.v10.jsonl', 'summary.v10.csv'
    return 'plan.json', 'invocations.jsonl', 'summary.csv'
