"""Explicit, immutable provenance for the reviewed v7 to ABC v8 migration.

Compatibility approves particular completed artifacts.  It never rewrites an
old fingerprint or treats every result sharing that fingerprint as compatible.
All functions are local and read only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

from .data import PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / 'configs/formula/compatibility/v7-small-to-abc-v8.json'
LEGACY_PROTOCOL = 'formula-abcd-compact-artifacts-bailian-v7'
SOURCE_CHANGE_ALLOWLIST = frozenset({
    'src/experiments/data.py', 'src/experiments/runner.py',
    'src/experiments/runtime.py', 'src/experiments/compatibility.py',
    'eval/finance/experiments.py',
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


def _check_result_metadata(result, group, trial, plan):
    _require(group in ('A', 'B', 'C') and isinstance(trial, int) and not isinstance(trial, bool),
             'invalid current group or trial')
    _require(1 <= trial <= plan['config']['repeats'], 'trial is outside the plan')
    _require(result.get('group') == group and type(result.get('trial')) is int
             and result['trial'] == trial, 'result group/trial does not match its folder')
    _require(result.get('status') == 'complete', 'only completed results are accepted')
    _require(type(result.get('mock')) is bool and result['mock'] == (plan['mode'] == 'mock'),
             'result execution mode does not match the plan')
    _require(type(result.get('formal_metrics')) is bool
             and result['formal_metrics'] == plan['formal_metrics'],
             'result formal-metrics provenance does not match the plan')


def _check_entry(root, key, entry, old_plan, target_plan):
    _require(isinstance(entry, dict), f'invalid preserved result entry: {key}')
    group, trial = entry.get('group'), entry.get('trial')
    _require(group in ARTIFACT_FILES and type(trial) is int and trial >= 1,
             'only explicitly listed historical A/B trials may be preserved')
    _require(key == f'{group}/{trial:02d}', 'preserved result path must match group/trial')
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
    _check_result_metadata(result, group, trial, target_plan)
    _require(result.get('fingerprint') == old_plan['fingerprint'], f'historical fingerprint changed: {key}')
    if group == 'B':
        book_sha = files['training/best_playbook.txt']
        _require(result.get('playbook_sha256') == book_sha, 'historical B playbook hash does not match result')
        training = _read_json(folder / 'training/train_results.json')
        best = training.get('best_checkpoint')
        if isinstance(best, dict) and 'playbook_sha256' in best:
            _require(best['playbook_sha256'] == book_sha, 'historical B best checkpoint hash does not match')
    return result


def load_compatibility(output_root, plan, manifest_path=None):
    """Validate an existing root plan and the exact reviewed historical files.

    Return a JSON-serializable provenance context for a legacy root; return
    ``None`` for an empty root or a matching native root.  No file is written.
    """
    root = Path(output_root)
    plan_path = root / 'plan.json'
    _require(not plan_path.is_symlink(), 'root plan must not be a symlink')
    _check_plan_digest(plan)
    if not plan_path.exists():
        return None
    existing = _read_json(plan_path)
    if existing.get('schema_version') == plan.get('schema_version'):
        _check_current_plan(existing, plan)
        return None
    _require(existing.get('schema_version') == 7 and existing.get('protocol') == LEGACY_PROTOCOL,
             'unrecognized historical plan protocol')
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    manifest = _read_json(path)
    _require(manifest.get('schema_version') == 1, 'unrecognized compatibility manifest schema')
    _require(isinstance(manifest.get('migration_id'), str) and bool(manifest['migration_id'].strip()),
             'compatibility migration_id must be a nonempty string')
    legacy, target = manifest.get('legacy'), manifest.get('target')
    _require(isinstance(legacy, dict) and isinstance(target, dict), 'legacy and target bindings are required')
    _require(legacy.get('plan_sha256') == _file_sha(plan_path), 'historical plan bytes changed')
    _check_plan_digest(existing)
    _require(all(legacy.get(field) == existing.get(field)
                 for field in ('schema_version', 'protocol', 'fingerprint')),
             'historical plan identity does not match the manifest')
    _require(plan.get('schema_version') == 8, 'migration target must use schema 8')
    _require(all(target.get(field) == plan.get(field) for field in PLAN_FIELDS),
             'target configuration, data, source, mode or protocol changed after review')
    converted = copy.deepcopy(existing['config'])
    groups = converted.get('groups')
    _require(isinstance(groups, dict) and set(groups) == set('ABCD'), 'historical groups must be ABCD')
    _require(groups['C'].get('kind') == 'ace' and groups['C'].get('max_num_rounds') == 1,
             'historical C must be the removed one-round arm')
    _require(groups['D'].get('kind') == 'jev_filter' and groups['D'].get('depends_on') == 'B',
             'historical D must be the B-dependent JEV arm')
    converted['groups'] = {'A': groups['A'], 'B': groups['B'], 'C': groups['D']}
    _require(converted == plan['config'], 'only removing old C and renaming D to C is compatible')
    _require(existing['data'] == plan['data'] and existing['mode'] == plan['mode']
             and existing.get('formal_metrics') == plan.get('formal_metrics'),
             'historical and target data/execution provenance must match')
    before, after = existing['source_sha256'], plan['source_sha256']
    changes = {name: {'old_sha256': before.get(name), 'new_sha256': after.get(name)}
               for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)}
    _require(set(changes) <= SOURCE_CHANGE_ALLOWLIST, 'unreviewed source file changed')
    _require(manifest.get('source_changes') == changes, 'source change review does not match the exact source diff')
    preserved = manifest.get('preserved_results')
    _require(isinstance(preserved, dict) and bool(preserved), 'explicit preserved artifact entries are required')
    for key, entry in preserved.items():
        _check_entry(root, key, entry, existing, plan)
    return {
        'migration_id': manifest['migration_id'],
        'manifest_path': str(path.resolve()), 'manifest_sha256': _file_sha(path),
        'legacy_plan_path': str(plan_path.resolve()), 'legacy_plan_sha256': _file_sha(plan_path),
        'legacy_fingerprint': existing['fingerprint'], 'target_fingerprint': plan['fingerprint'],
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
    _require(fingerprint == context['legacy_fingerprint'] == entry['original_fingerprint'],
             'historical result fingerprint does not match its preserved origin')
    return {'kind': 'legacy_compatible', 'fingerprint': fingerprint,
            'result_sha256': _file_sha(path), 'manifest_sha256': context['manifest_sha256'],
            'legacy_plan_sha256': context['legacy_plan_sha256'],
            'migration_id': context['migration_id'], 'manifest_path': context['manifest_path']}


def storage_names(output_root, plan, compatibility):
    """Use distinct root bookkeeping files when preserving a legacy plan."""
    if compatibility is not None:
        _require(compatibility.get('target_fingerprint') == plan.get('fingerprint'),
                 'storage context target does not match the plan')
        return 'plan.v8.json', 'invocations.v8.jsonl', 'summary.v8.csv'
    return 'plan.json', 'invocations.jsonl', 'summary.csv'
