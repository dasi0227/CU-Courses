"""Paired Formula trials with explicit model execution and saved results."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import importlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from .data import GROUPS, PROJECT_ROOT, load_config, prepare_data, validate_llm_config, validate_config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def build_plan(config, data):
    validate_llm_config(config['llm'])
    count = len(data['train'])
    if count % config['training']['eval_steps']:
        raise ValueError('eval_steps must divide the training count so every epoch ends with validation')
    return {
        'config': config,
        'ids': {split: [sample['id'] for sample in data[split]]
                for split in ('train', 'val', 'test')},
    }


def processed_samples(raw, processor):
    processed = processor.process_task_data(raw)
    for source, row in zip(raw, processed):
        row.update(id=source['id'], formula_name=source['formula_name'])
    return processed


class SelectedGenerator:
    def __init__(self, generator, selector, stage):
        self.generator, self.selector, self.stage = generator, selector, stage

    def generate(self, **kwargs):
        selected, selection = self.selector.select(
            kwargs['playbook'], kwargs['question'],
            call_id=f"{self.stage}:{kwargs.get('call_id', 'gen')}", stage=self.stage,
        )
        kwargs['playbook'] = selected
        response, bullets, info = self.generator.generate(**kwargs)
        return response, bullets, {**info, 'selection': selection}


def jev_client(config, output):
    from .jev import JevClient
    return JevClient(
        model=config['model'], base_url=config['base_url'],
        api_key_env=config['api_key_env'],
        enabled=True, audit_path=output / 'jev_audit.jsonl', timeout=config['timeout'],
    )


def jev_summary(client):
    events = [event for event in client.events if event.get('event') == 'jev_request'] if client else []
    return {'attempts': len(events),
            'successful_calls': sum(event.get('status') == 'success' for event in events),
            'latency_seconds': sum(event.get('latency_seconds', 0) for event in events),
            'reported_usage': [event.get('usage', {}) for event in events]}


def checked_evaluation(processor, generator, book, samples, runtime, output, stage, workers):
    from src.common.utils import evaluate_test_set
    previous = runtime.stage
    runtime.stage = stage
    try:
        result, errors = evaluate_test_set(
            processor, generator, book, samples, runtime.config['max_tokens'],
            None, max_workers=workers, use_json_mode=False, stage=stage,
            predictions_path=output / 'predictions.jsonl' if stage == 'test' else None,
            compact_predictions=True,
        )
        record_evaluation_errors(runtime, result)
    finally:
        runtime.stage = previous
    return {key: value for key, value in result.items() if key != 'predictions'}


def record_evaluation_errors(runtime, metrics):
    for row in metrics['predictions']:
        if row.get('error'):
            runtime.record_error(kind='evaluation', role='generator',
                                 call_id=row['call_id'], stage=row['stage'],
                                 sample_id=row['sample_id'],
                                 error_type='AnswerParseError' if row['api_success'] else 'EvaluationError',
                                 message=row['error'])


def trial_folder(output_root, group, trial):
    return Path(output_root) / group / f'{trial:02d}'


def check_existing_plan(output_root, plan):
    path = Path(output_root) / 'plan.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        fields = ('config', 'ids')
        if any(saved.get(field) != plan.get(field) for field in fields):
            raise ValueError('Output directory belongs to a different configuration or selected data')


def check_completed_result(result, group, trial):
    if (result.get('group') != group or type(result.get('trial')) is not int
            or result['trial'] != trial):
        raise ValueError('Result group/trial does not match its folder')
    if result.get('status') != 'complete':
        raise ValueError('A completed result is required')
    if result.get('mock') is not False:
        raise ValueError('A real model result is required')


def b_checkpoint(output_root, trial, plan):
    if not isinstance(plan, dict):
        raise ValueError('A complete plan is required to read a B checkpoint')
    check_existing_plan(output_root, plan)
    folder = trial_folder(output_root, 'B', trial)
    result_path = folder / 'result.json'
    if not result_path.exists():
        raise ValueError(f'C requires an already completed B trial: {result_path}')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('group') != 'B' or result.get('trial') != trial:
        raise ValueError('C requires a B checkpoint from the same trial')
    if result.get('status') != 'complete':
        raise ValueError('C requires a completed B trial')
    check_completed_result(result, 'B', trial)
    path = folder / 'training' / 'best_playbook.txt'
    text = path.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError('B best playbook is empty')
    learning_stages = [usage for stage, usage in result['llm']['stage_usage'].items() if stage != 'test']
    learning_usage = {key: sum(stage[key] for stage in learning_stages)
                      for key in ('successful_calls', 'attempts', 'prompt_tokens', 'completion_tokens')}
    return text, {'group': 'B', 'trial': trial, 'path': str(path),
                  'training_cost_included_in_C_incremental_usage': False,
                  'shared_learning_llm_usage': learning_usage,
                  'shared_learning_excludes_B_test': True}


def run_group(group, trial, plan, data, output_root):
    validate_config(plan['config'])
    if group not in GROUPS:
        raise ValueError('Only A/B/C groups can run')
    if type(trial) is not int or not 1 <= trial <= plan['config']['repeats']:
        raise ValueError('trial must be an integer inside 1..repeats')
    check_existing_plan(output_root, plan)
    inherited_book, lineage = b_checkpoint(output_root, trial, plan) if group == 'C' else (None, None)
    from src import ACE
    from eval.data_processor import DataProcessor
    from .runtime import Runtime
    from .jev import ReadSelector

    config = plan['config']
    folder = trial_folder(output_root, group, trial)
    folder.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(config['llm'], folder)
    result = {'group': group, 'trial': trial, 'status': 'running', 'mock': False,
              'predictions_path': str(folder / 'predictions.jsonl')}
    write_json(folder / 'result.json', result)
    processor = DataProcessor('formula')
    selected = {split: processed_samples(data[split], processor)
                for split in ('train', 'val', 'test')}
    workers = config['training'].get('test_workers', 4)
    client = None
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            stack.enter_context(runtime.installed())
            engine = ACE(api_provider=config['llm']['api_provider'],
                         generator_model=config['llm']['model'], reflector_model=config['llm']['model'],
                         curator_model=config['llm']['model'], max_tokens=config['llm']['max_tokens'],
                         use_bulletpoint_analyzer=False)
            book = engine.playbook
            generator = engine.generator
            if group == 'C':
                client = jev_client(config['jev'], folder)
            if group == 'B':
                training_dir = folder / 'training'
                def fixed_paths(save_dir, task_name, mode):
                    training_dir.mkdir(parents=True, exist_ok=True)
                    return str(training_dir), None, None, str(training_dir)
                engine._setup_paths = fixed_paths
                # Route training validation through the same fixed-denominator evaluator.
                ace_module = importlib.import_module('src.ace')
                from src.common.utils import evaluate_test_set
                def validation(*args, **kwargs):
                    previous = runtime.stage
                    runtime.stage = kwargs.get('stage', 'validation')
                    kwargs['compact_predictions'] = True
                    try:
                        metrics, errors = evaluate_test_set(*args, **kwargs)
                        record_evaluation_errors(runtime, metrics)
                    finally:
                        runtime.stage = previous
                    return metrics, errors
                stack.enter_context(patch.object(ace_module, 'evaluate_test_set', validation))
                runtime.stage = 'training'
                ace_config = {**config['training'],
                              'task_name': 'formula', 'save_dir': str(training_dir), 'test_workers': workers,
                              'json_mode': False, 'no_ground_truth': False,
                              'playbook_token_budget': 80000, 'use_bulletpoint_analyzer': False,
                              'compact_artifacts': True}
                result['training_config'] = {
                    'task_name': 'formula', 'mode': 'offline',
                    'generator_model': config['llm']['model'],
                    'reflector_model': config['llm']['model'],
                    'curator_model': config['llm']['model'],
                    'config': ace_config,
                    'effective_parameters': engine._extract_config_params(ace_config),
                }
                result['training_results_path'] = str(training_dir / 'train_results.json')
                write_json(folder / 'result.json', result)
                training = engine.run('offline', selected['train'], selected['val'], [], processor, ace_config)
                result['training'] = training
                book = engine.best_playbook
            if group == 'C':
                book = inherited_book
                result['b_checkpoint'] = lineage
                k = config['jev']['k']
                result.update(selection_k=k, selection_method='fixed_top_k',
                              selection_primitive='score',
                              selection_ranking='score_desc_then_playbook_order',
                              selection_confidence_used_for_ranking=False)
                generator = SelectedGenerator(generator, ReadSelector(client, k, config['jev']['batch_size']), 'test')
            result['playbook_source'] = (
                str(folder / 'training' / 'best_playbook.txt') if group == 'B' else
                result['b_checkpoint']['path'] if group == 'C' else 'empty_playbook'
            )
            # Persist the fixed selection policy and checkpoint lineage before
            # testing, including when a run is interrupted during evaluation.
            write_json(folder / 'result.json', result)
            metrics = checked_evaluation(processor, generator, book, selected['test'], runtime, folder, 'test', workers)
            result['test'] = metrics
            if metrics['api_failure_count']:
                raise RuntimeError('test: API failures occurred; metrics retained, trial is incomplete')
            result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=runtime._safe(str(exc)))
        runtime.record_error(kind='run', role='runner', call_id=f'{group}:{trial:02d}',
                             error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        result.update(wall_seconds=time.monotonic() - started, llm=runtime.summary(),
                      jev=jev_summary(client), errors=runtime.errors)
        shared = result.get('b_checkpoint', {}).get('shared_learning_llm_usage', {})
        result['llm_usage_including_shared_learning'] = {
            key: result['llm'][key] + shared.get(key, 0)
            for key in ('successful_calls', 'attempts', 'prompt_tokens', 'completion_tokens')}
        write_json(folder / 'result.json', result)
        if runtime._client is not None:
            runtime._client.close()
    return result


def aggregate(output_root, plan):
    output_root = Path(output_root)
    check_existing_plan(output_root, plan)
    collected = []
    for group in GROUPS:
        for trial in range(1, plan['config']['repeats'] + 1):
            path = trial_folder(output_root, group, trial) / 'result.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding='utf-8'))
            if row.get('status') != 'complete':
                continue
            check_completed_result(row, group, trial)
            if group == 'C' and (
                row.get('selection_k') != plan['config']['jev']['k']
                or row.get('selection_primitive') != 'score'
                or row.get('selection_ranking') != 'score_desc_then_playbook_order'
                or row.get('selection_confidence_used_for_ranking') is not False
            ):
                raise ValueError('C result selection policy does not match the current Score top-10 plan')
            if group == 'C':
                checkpoint = row.get('b_checkpoint', {})
                if (checkpoint.get('group') != 'B' or type(checkpoint.get('trial')) is not int
                        or checkpoint['trial'] != trial):
                    raise ValueError('C requires a B checkpoint from the same trial')
                b_checkpoint(output_root, trial, plan)
            collected.append(row)
    summaries = []
    for group in GROUPS:
        rows = [row for row in collected if row['group'] == group]
        if not rows:
            continue
        accuracies = [row['test']['accuracy'] for row in rows]
        summaries.append({'group': group, 'completed_trials': len(rows), 'expected_trials': plan['config']['repeats'],
                          'selection_k': plan['config']['jev']['k'] if group == 'C' else None,
                          'accuracy_mean': statistics.mean(accuracies),
                          'accuracy_sample_variance': statistics.variance(accuracies) if len(rows) > 1 else None,
                          'accuracy_sample_std': statistics.stdev(accuracies) if len(rows) > 1 else None,
                          'mean_wall_seconds': statistics.mean(row['wall_seconds'] for row in rows),
                          'mean_llm_calls_incremental': statistics.mean(row['llm']['successful_calls'] for row in rows),
                          'mean_llm_calls_including_shared_learning': statistics.mean(
                              row['llm_usage_including_shared_learning']['successful_calls'] for row in rows),
                          'mean_prompt_tokens_incremental': statistics.mean(row['llm']['prompt_tokens'] for row in rows),
                          'mean_prompt_tokens_including_shared_learning': statistics.mean(
                              row['llm_usage_including_shared_learning']['prompt_tokens'] for row in rows),
                          'mean_completion_tokens_incremental': statistics.mean(row['llm']['completion_tokens'] for row in rows),
                          'mean_completion_tokens_including_shared_learning': statistics.mean(
                              row['llm_usage_including_shared_learning']['completion_tokens'] for row in rows),
                          'mean_jev_calls': statistics.mean(row['jev']['successful_calls'] for row in rows),
                          'mock': False})
    if summaries:
        with (output_root / 'summary.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


class ExperimentHelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if action.dest == 'groups':
            return '--groups GROUP'
        return super()._format_action_invocation(action)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=ExperimentHelpFormatter)
    parser.add_argument('-h', '--help', action='help', help='Show help and exit.')
    parser.add_argument('--config', metavar='PATH', help='Config file (default: config.json).')
    parser.add_argument('--groups', nargs='+', choices=GROUPS, metavar='GROUP',
                        help='Select A, B, C (default: all).')
    parser.add_argument('--trial', type=int, metavar='N', help='Trial number (default: all).')
    parser.add_argument('--output', type=Path, metavar='DIR', help='Results directory (default: results).')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true', help='Run experiments using real APIs.')
    modes.add_argument('--dry-run', action='store_true', help='Create/check the plan only (default).')
    args = parser.parse_args(argv)
    config = load_config(args.config or 'config.json')
    groups = [group for group in GROUPS if group in (args.groups or GROUPS)]
    if not groups:
        parser.error('No groups selected')
    required_keys = [config['llm']['api_key_env']]
    if 'C' in groups:
        required_keys.append(config['jev']['api_key_env'])
    trials = [args.trial] if args.trial is not None else list(range(1, config['repeats'] + 1))
    if any(trial < 1 or trial > config['repeats'] for trial in trials):
        parser.error('--trial is outside 1..repeats')
    data = prepare_data(config, PROJECT_ROOT)
    plan = build_plan(config, data)
    default_output = PROJECT_ROOT / config['output_dir']
    output = (args.output or default_output).resolve()
    try:
        check_existing_plan(output, plan)
    except ValueError as exc:
        parser.error(str(exc))
    plan_path = output / 'plan.json'
    if args.execute:
        # Preflight every selected output/dependency before any paid request.
        for trial in trials:
            for group in groups:
                if trial_folder(output, group, trial).exists():
                    parser.error(f'Output already exists for trial {trial} group {group}; use a new output directory')
            if 'C' in groups and 'B' not in groups:
                b_checkpoint(output, trial, plan)
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / '.env')
        missing = [name for name in required_keys if not os.environ.get(name, '').strip()]
        if missing:
            parser.error('Missing keys: ' + ', '.join(missing))
    output.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        write_json(plan_path, plan)
    print(json.dumps({'mode': 'execute' if args.execute else 'dry-run',
                      'groups': groups, 'trials': trials,
                      'samples': {split: len(data[split]) for split in ('train', 'val', 'test')},
                      'llm_model': config['llm']['model'],
                      'llm_provider': config['llm']['api_provider'],
                      'llm_base_url': config['llm']['base_url'],
                      'enable_thinking': config['llm']['enable_thinking'],
                      'jev_model': config['jev']['model'] if 'C' in groups else None,
                      'required_keys_for_live_run': required_keys, 'output': str(output),
                      'plan_path': str(plan_path)}, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    try:
        for trial in trials:
            for group in groups:
                run_group(group, trial, plan, data, output)
                aggregate(output, plan)
    except Exception as exc:
        aggregate(output, plan)
        print(f'Run stopped: {type(exc).__name__}: {exc}')
        return 1
    return 0
