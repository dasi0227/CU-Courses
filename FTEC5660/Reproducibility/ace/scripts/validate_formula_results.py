#!/usr/bin/env python3
"""Read-only, credential-free validation of the persisted Formula results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.compatibility import (  # noqa: E402
    load_compatibility, read_provenance_document, storage_names, validate_result_origin,
)
from src.experiments.data import GROUPS, load_config, prepare_data  # noqa: E402
from src.experiments.runner import build_plan, digest  # noqa: E402


NO_ANSWER = 'No final answer found'
USAGE_FIELDS = ('successful_calls', 'prompt_tokens', 'completion_tokens')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f'{path}: records must be objects')
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label):
    require(type(value) is int and value >= 0, f'{label}: expected a nonnegative integer')
    return value


def number_equal(actual, expected, label):
    require(not isinstance(actual, bool) and isinstance(actual, (int, float))
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f'{label}: expected {expected}, got {actual}')


def answer_is_correct(prediction, target):
    """The existing Formula numeric exact-match rule, without importing API helpers."""
    try:
        return float(prediction.replace(',', '')) == float(target.replace(',', ''))
    except (AttributeError, ValueError, TypeError):
        return prediction == target


def validate_trial(folder, group, trial, samples, llm_config):
    """Validate one completed trial; return its unchanged result dictionary."""
    folder = Path(folder)
    result = read_json(folder / 'result.json')
    prefix = f'{group}/{trial:02d}'
    require(result.get('group') == group and result.get('trial') == trial
            and result.get('status') == 'complete', f'{prefix}: result identity/status mismatch')
    predictions = read_jsonl(folder / 'predictions.jsonl')
    ids = [row['id'] for row in samples]
    require(len(ids) > 0 and len(set(ids)) == len(ids), 'selected test IDs must be nonempty and unique')
    require([row.get('sample_id') for row in predictions] == ids,
            f'{prefix}: prediction IDs/count/order differ from the selected test set')
    correct = api_success = no_answer = 0
    for index, (row, sample) in enumerate(zip(predictions, samples)):
        label = f'{prefix}: prediction {index}'
        require(row.get('index') == index and row.get('call_id') == f'test_eval_{index}'
                and row.get('stage') == 'test', f'{label}: identity mismatch')
        require(row.get('target') == sample['target'], f'{label}: target changed')
        require(isinstance(row.get('prediction'), str), f'{label}: prediction must be a string')
        require(type(row.get('api_success')) is bool and type(row.get('is_correct')) is bool,
                f'{label}: outcome flags must be booleans')
        expected_correct = answer_is_correct(row['prediction'], sample['target'])
        require(row['is_correct'] == expected_correct, f'{label}: incorrect exact-match flag')
        require(row['api_success'] or not row['is_correct'], f'{label}: API failure cannot be correct')
        correct += row['is_correct']
        api_success += row['api_success']
        no_answer += row['prediction'] == NO_ANSWER
    expected_metrics = {
        'total': len(samples), 'correct': correct, 'accuracy': correct / len(samples),
        'no_answer': no_answer, 'api_success_count': api_success,
        'api_failure_count': len(samples) - api_success, 'success_rate': api_success / len(samples),
    }
    for key, value in expected_metrics.items():
        number_equal(result['test'].get(key), value, f'{prefix}: test.{key}')
    require(result['test'].get('stage') == 'test', f'{prefix}: test stage mismatch')
    wall = result.get('wall_seconds')
    require(not isinstance(wall, bool) and isinstance(wall, (int, float))
            and math.isfinite(wall) and wall >= 0, f'{prefix}: invalid cumulative elapsed time')
    for key in USAGE_FIELDS:
        integer(result['llm'].get(key), f'{prefix}: llm.{key}')
    integer(result['llm'].get('attempts'), f'{prefix}: llm.attempts')
    require(result['llm']['attempts'] >= result['llm']['successful_calls'],
            f'{prefix}: cumulative requests smaller than successful responses')

    if result.get('artifact_view') == 'final_predictions':
        require(group == 'A' and trial in (2, 3), f'{prefix}: unexpected final-prediction view')
        require(not {'recovery', 'format_recovery', 'errors'}.intersection(result),
                f'{prefix}: final-prediction view contains historical records')
        calls = read_jsonl(folder / 'llm_calls.jsonl')
        require(len(calls) == len(samples), f'{prefix}: exactly one final response is required per sample')
        require({call.get('call_id') for call in calls} == {row['call_id'] for row in predictions},
                f'{prefix}: final response call IDs are missing, duplicated, or unrelated')
        totals = dict.fromkeys(USAGE_FIELDS, 0)
        for index, call in enumerate(calls):
            label = f'{prefix}: final response {index}'
            require(not {'attempt', 'recovery', 'format_recovery', 'transport_timeout_seconds',
                         'error', 'error_message', 'event'}.intersection(call),
                    f'{label}: contains transport/failure history')
            require(call.get('stage') == 'test' and call.get('role') == 'generator'
                    and call.get('api_success') is True, f'{label}: final response identity/status mismatch')
            require(isinstance(call.get('response'), str) and call['response'].strip(),
                    f'{label}: missing final response')
            require(isinstance(call.get('prompt'), str) and call['prompt'].strip(),
                    f'{label}: missing original prompt')
            for key in ('model', 'api_provider', 'base_url', 'temperature', 'max_tokens', 'enable_thinking'):
                require(call.get(key) == llm_config[key], f'{label}: changed model setting {key}')
            totals['successful_calls'] += 1
            totals['prompt_tokens'] += integer(call.get('prompt_num_tokens'), f'{label}: input tokens')
            totals['completion_tokens'] += integer(call.get('response_num_tokens'), f'{label}: output tokens')
        require(result.get('final_llm_usage') == totals, f'{prefix}: final response token/call totals mismatch')
        for key in USAGE_FIELDS:
            require(result['llm'][key] >= totals[key],
                    f'{prefix}: cumulative usage undercounts final response {key}')
    return result


def checked_plan(path, expected_sha=None, expected_fingerprint=None, *, compatibility=None):
    if expected_sha is None:
        plan = read_json(path)
    else:
        plan = read_provenance_document(compatibility, path, expected_sha)
    expected = digest({'config': plan['config'], 'data': plan['data'],
                       'source': plan['source_sha256'], 'mock': plan['mode'] == 'mock'})
    require(plan.get('fingerprint') == expected, f'{path}: historical plan fingerprint is inconsistent')
    if expected_fingerprint is not None:
        require(plan['fingerprint'] == expected_fingerprint, f'{path}: historical plan fingerprint changed')
    return plan


def validate_historical_reference(reference, compatibility=None):
    """Resolve original provenance from the unified plan, never from superseded files."""
    if not reference or 'manifest_path' not in reference:
        return
    manifest_path = reference['manifest_path']
    manifest = read_provenance_document(compatibility, manifest_path, reference.get('manifest_sha256'))
    require(manifest.get('migration_id') == reference.get('migration_id'),
            f'{manifest_path}: recorded migration ID changed')
    if 'target_fingerprint' in reference:
        require(manifest['target']['fingerprint'] == reference['target_fingerprint'],
                f'{manifest_path}: recorded target fingerprint changed')
    if 'legacy_plan_path' in reference:
        checked_plan(reference['legacy_plan_path'], reference['legacy_plan_sha256'],
                     reference.get('legacy_fingerprint'), compatibility=compatibility)
    for source_id, source in reference.get('origin_plans', {}).items():
        recorded = manifest['origin_plans'][source_id]
        for key in ('path', 'plan_sha256', 'schema_version', 'protocol', 'fingerprint'):
            require(source.get(key) == recorded.get(key), f'{source_id}: origin plan binding changed')
        checked_plan(source['resolved_path'], source['plan_sha256'], source['fingerprint'],
                     compatibility=compatibility)


def validate_checkpoint(output_root, trial, b_result, c_result, compatibility=None):
    """Check C's same-number handbook and its original, unchanged B provenance."""
    output_root = Path(output_root)
    book_path = output_root / 'B' / f'{trial:02d}' / 'training' / 'best_playbook.txt'
    lineage = c_result['b_checkpoint']
    require(lineage.get('group') == 'B' and lineage.get('trial') == trial,
            f'C/{trial:02d}: checkpoint is not from the matching B trial')
    require(Path(lineage['path']).resolve() == book_path.resolve(),
            f'C/{trial:02d}: recorded B playbook path changed')
    book_sha = sha256(book_path)
    require(book_sha == lineage.get('sha256') == b_result.get('playbook_sha256')
            == c_result.get('playbook_sha256'), f'C/{trial:02d}: B playbook SHA-256 mismatch')
    require(lineage.get('original_fingerprint') == b_result.get('fingerprint'),
            f'C/{trial:02d}: recorded B source fingerprint changed')
    origin = lineage['origin']
    require(origin.get('fingerprint') == b_result['fingerprint']
            and origin.get('result_sha256') == sha256(book_path.parents[1] / 'result.json'),
            f'C/{trial:02d}: recorded B result identity/SHA-256 changed')
    require(lineage.get('shared_learning_excludes_B_test') is True
            and lineage.get('training_cost_included_in_C_incremental_usage') is False,
            f'C/{trial:02d}: shared learning accounting policy changed')
    stages = [value for stage, value in b_result['llm']['stage_usage'].items() if stage != 'test']
    expected_shared = {key: sum(stage[key] for stage in stages)
                       for key in (*USAGE_FIELDS, 'attempts')}
    require(lineage.get('shared_learning_llm_usage') == expected_shared,
            f'C/{trial:02d}: shared B learning usage mismatch')
    for key in (*USAGE_FIELDS, 'attempts'):
        require(c_result['llm_usage_including_shared_learning'][key]
                == c_result['llm'][key] + expected_shared[key],
                f'C/{trial:02d}: combined learning usage mismatch for {key}')
    if origin.get('kind') == 'legacy_compatible':
        validate_historical_reference(origin, compatibility)
        manifest = read_provenance_document(compatibility, origin['manifest_path'], origin['manifest_sha256'])
        entry = manifest['preserved_results'][f'B/{trial:02d}']
        require(entry.get('original_fingerprint') == b_result['fingerprint']
                and entry['files_sha256']['result.json'] == origin['result_sha256'],
                f'C/{trial:02d}: original manifest does not bind this B result')
        source = manifest['origin_plans'][origin['source_plan_id']]
        require(source['fingerprint'] == origin.get('source_plan_fingerprint')
                == b_result['fingerprint'] and source['plan_sha256'] == origin.get('source_plan_sha256'),
                f'C/{trial:02d}: original B source plan binding changed')
        checked_plan(origin['source_plan_path'], source['plan_sha256'], source['fingerprint'],
                     compatibility=compatibility)
    else:
        require(origin.get('kind') == 'current', f'C/{trial:02d}: unsupported recorded B origin')
    require(c_result.get('selection_k') == 10 and c_result.get('selection_primitive') == 'score'
            and c_result.get('selection_ranking') == 'score_desc_then_playbook_order'
            and c_result.get('selection_confidence_used_for_ranking') is False,
            f'C/{trial:02d}: Score top-10 selection policy changed')


def validate_summary(path, results):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        summaries = list(csv.DictReader(handle))
    require([row['group'] for row in summaries] == list(GROUPS), 'summary group identities/order mismatch')
    for summary in summaries:
        group = summary['group']
        rows = [result for (name, _), result in results.items() if name == group]
        accuracies = [row['test']['accuracy'] for row in rows]
        expected = {
            'completed_trials': len(rows), 'expected_trials': len(rows),
            'accuracy_mean': statistics.mean(accuracies),
            'accuracy_sample_variance': statistics.variance(accuracies),
            'accuracy_sample_std': statistics.stdev(accuracies),
            'mean_wall_seconds': statistics.mean(row['wall_seconds'] for row in rows),
            'mean_llm_calls_incremental': statistics.mean(row['llm']['successful_calls'] for row in rows),
            'mean_llm_calls_including_shared_learning': statistics.mean(
                row['llm_usage_including_shared_learning']['successful_calls'] for row in rows),
            'mean_jev_calls': statistics.mean(row['jev']['successful_calls'] for row in rows),
        }
        for kind in ('prompt_tokens', 'completion_tokens'):
            expected[f'mean_{kind}_incremental'] = statistics.mean(row['llm'][kind] for row in rows)
            expected[f'mean_{kind}_including_shared_learning'] = statistics.mean(
                row['llm_usage_including_shared_learning'][kind] for row in rows)
        for key, value in expected.items():
            number_equal(float(summary[key]), value, f'summary {group}.{key}')


def validate_layout(output_root, repeats):
    """Require the single-plan result layout, without following substituted links."""
    output_root = Path(output_root)
    require(output_root.is_dir() and not output_root.is_symlink(),
            'results root must be a plain directory')
    expected = {'plan.json', 'summary.csv', *GROUPS}
    require({path.name for path in output_root.iterdir()} == expected,
            'results root must contain only plan.json, summary.csv, A, B and C')
    for name in expected:
        path = output_root / name
        require(not path.is_symlink(), f'results/{name}: symlink is not an accepted artifact')
        require(path.is_dir() if name in GROUPS else path.is_file(),
                f'results/{name}: unexpected file/directory type')
    expected_trials = {f'{trial:02d}' for trial in range(1, repeats + 1)}
    for group in GROUPS:
        folder = output_root / group
        require({path.name for path in folder.iterdir()} == expected_trials,
                f'{group}: trial directory set differs from the selected repetitions')
        require(all(path.is_dir() and not path.is_symlink() for path in folder.iterdir()),
                f'{group}: trial paths must be plain directories')


def validate_results(output_root, config_path):
    output_root = Path(output_root).absolute()
    config = load_config(config_path)
    validate_layout(output_root, config['repeats'])
    output_root = output_root.resolve()
    data = prepare_data(config, ROOT)
    plan = build_plan(config, data, list(GROUPS))
    compatibility = load_compatibility(output_root, plan)
    plan_name, summary_name = storage_names(output_root, plan, compatibility)
    recorded_plan = checked_plan(output_root / plan_name)
    for key in ('schema_version', 'protocol', 'fingerprint', 'config', 'data', 'source_sha256', 'mode', 'formal_metrics'):
        require(recorded_plan[key] == plan[key], f'current plan.{key} differs from the local checked source/configuration')
    results = {}
    for group in GROUPS:
        for trial in range(1, config['repeats'] + 1):
            folder = output_root / group / f'{trial:02d}'
            result = validate_trial(folder, group, trial, data['test'], config['llm'])
            validate_result_origin(output_root, group, trial, plan, result, compatibility=compatibility)
            validate_historical_reference(result.get('compatibility_source'), compatibility)
            results[group, trial] = result
    for trial in range(1, config['repeats'] + 1):
        validate_checkpoint(output_root, trial, results['B', trial], results['C', trial], compatibility)
    validate_summary(output_root / summary_name, results)
    return {'validated_trials': len(results), 'test_samples_per_trial': len(data['test']),
            'plan': str(output_root / plan_name), 'summary': str(output_root / summary_name),
            'accuracy': {f'{group}/{trial:02d}': result['test']['accuracy']
                         for (group, trial), result in results.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/formula/small.json')
    args = parser.parse_args(argv)
    print(json.dumps(validate_results(args.output, args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
