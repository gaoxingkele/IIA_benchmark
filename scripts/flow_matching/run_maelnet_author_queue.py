"""Run complete pinned author recipes in separate seed workspaces; retain failures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def argument(arguments, name, default=None):
    flag = '--' + name
    return arguments[arguments.index(flag) + 1] if flag in arguments else default


def replace_argument(arguments, name, value):
    result = list(arguments)
    flag = '--' + name
    if flag in result:
        index = result.index(flag)
        result[index + 1] = str(value)
    else:
        result.extend([flag, str(value)])
    return result


def effective_arguments(stage, job, settings, workspace):
    arguments = list(stage['author_arguments'])
    for name, value in {'root_path': ROOT / job['raw_directory'], 'result_dir': workspace / 'results',
                        'checkpoints': './checkpoints', 'seed': job['seed'],
                        'num_workers': settings['num_workers'], 'chunk_size': settings['chunk_size']}.items():
        arguments = replace_argument(arguments, name, value)
    return arguments


def verify_artifacts(receipt, root=ROOT):
    if receipt['status'] != 'completed' or not receipt.get('artifacts'):
        raise ValueError('No complete stage receipt')
    for record in receipt['artifacts']:
        if sha(root / record['path']) != record['sha256']:
            raise ValueError('Stage artifact changed: ' + record['path'])


def training_rows(csv_path):
    with Path(csv_path).open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.reader(stream) if len(r) == 6 and r[0].isdigit()]
    if not rows or [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Missing/nonconsecutive author epoch records')
    values = [[float(v) for v in row] for row in rows]
    if not all(math.isfinite(v) for row in values for v in row):
        raise ValueError('Nonfinite author training receipt')
    return values


def stage_artifacts(workspace, before, arguments, data_audit, stdout):
    is_training = int(argument(arguments, 'is_training', 1)) == 1
    checkpoints = list((workspace / 'checkpoints').rglob('*.pth'))
    if is_training:
        slow = argument(arguments, 'is_slow_learner') is not None
        model = argument(arguments, 'model', 'AnomalyTransformer')
        filename = 'checkpoint_' + ('slow_learner_' if slow else '') + model + '.pth'
        matches = [p for p in checkpoints if p.name == filename]
        csvs = [p for p in (workspace / 'results').glob('training*.csv') if str(p) not in before]
        if len(matches) != 1 or len(csvs) != 1:
            raise ValueError('Expected exactly one author checkpoint and epoch CSV')
        rows = training_rows(csvs[0])
        maximum = int(argument(arguments, 'train_epochs', 10))
        early_stopped = 'Early stopping' in stdout.read_text(encoding='utf-8', errors='replace')
        expected_steps = math.ceil(data_audit['window_counts']['train'] / int(argument(arguments, 'batch_size', 32)))
        if len(rows) > maximum or (len(rows) < maximum and not early_stopped):
            raise ValueError('Author epoch budget ended without early stopping')
        if any(int(row[2]) != expected_steps for row in rows):
            raise ValueError('Training does not cover all author windows')
        paths = matches + csvs
        import torch
        weights = torch.load(matches[0], map_location='cpu', weights_only=True)
        if not isinstance(weights, dict) or not weights or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in weights.values()):
            raise ValueError('Invalid/nonfinite learner checkpoint')
        details = {'model': model, 'epochs_completed': len(rows), 'maximum_epochs': maximum,
                   'author_early_stopping': early_stopped, 'steps_per_epoch': expected_steps,
                   'training_windows': data_audit['window_counts']['train'], 'epoch_rows': rows,
                   'checkpoint_tensor_elements_including_buffers': sum(v.numel() for v in weights.values())}
    else:
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        required = ['author_rl_inputs.npz', 'author_rl_predictions.npz', 'author_rl_metrics.json', 'dqn_policy.zip']
        paths = []
        for name in required:
            matches = list((workspace / 'logs').rglob(name))
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous author RL artifact: ' + name)
            paths.extend(matches)
        if len(checkpoints) != 3:
            raise ValueError('The original ensemble requires three learner checkpoints')
        paths.extend(checkpoints)
        metrics = read(paths[2])
        expected = data_audit['window_counts']['test'] * data_audit['window']
        with np.load(paths[0]) as inputs, np.load(paths[1]) as predictions:
            labels, scores = inputs['labels'], inputs['model_scores']
            predicted = predictions['predictions']
            if scores.shape != (3, expected) or labels.shape != (expected,) or predicted.shape != labels.shape:
                raise ValueError('Incomplete author flattened-window score coverage')
            if not np.isfinite(scores).all() or not np.isfinite(inputs['thresholds']).all():
                raise ValueError('Nonfinite author scores')
            if not np.array_equal(labels, predictions['labels']) or not np.isin(labels, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
                raise ValueError('Mismatched/nonbinary author labels/predictions')
            p, r, f, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
            independent = {'accuracy': accuracy_score(labels, predicted), 'precision': p, 'recall': r, 'f1': f}
            for key, value in independent.items():
                if not math.isclose(metrics[key], value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Author RL reported metric differs: ' + key)
        if metrics['label_points'] != expected or not expected <= metrics['policy_timesteps'] < expected + 4:
            raise ValueError('DQN budget does not cover the complete author test sequence')
        details = {'author_rl_metrics': metrics, 'flattened_test_points': expected,
                   'native_test_points': data_audit['shapes']['test'][0], 'test_labels_used_for_learning': True,
                   'PA_derived_environment_states': True, 'independent_test': False}
    return [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths], details


def run_job(job, settings, state, state_path):
    base = ROOT / job['output_directory']
    base.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    final = base / 'result.json'
    if final.exists():
        receipt = read(final)
        if receipt['experiment_sha256'] != fingerprint:
            raise ValueError('Job identity changed')
        verify_artifacts(receipt)
        return 'completed'
    workspace = base / 'author_workspace'
    if not workspace.exists():
        shutil.copytree(ROOT / settings['corrected_source'], workspace, ignore=shutil.ignore_patterns('__pycache__'))
    manifest = read(ROOT / settings['corrected_source'] / 'runtime_patch_manifest.json')
    for file in manifest['files']:
        if sha(workspace / file['path']) != file['corrected_sha256']:
            raise ValueError('Private author workspace changed')
    (workspace / 'results').mkdir(exist_ok=True)
    data_audit = read(ROOT / settings['preflight_report'])['native_datasets'][job['dataset']]
    receipts = []
    environment = {key.upper(): value for key, value in os.environ.items()}
    environment.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS=str(settings['cpu_threads']),
                       MKL_NUM_THREADS=str(settings['cpu_threads']), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    for stage in job['stages']:
        stage_dir = base / 'stages' / stage['id']
        stage_receipt = stage_dir / 'receipt.json'
        if stage_receipt.exists():
            record = read(stage_receipt)
            if record['stage_fingerprint'] != fingerprint:
                raise ValueError('Stage receipt belongs to another frozen job')
            verify_artifacts(record)
            receipts.append(record)
            continue
        if stage_dir.exists():
            raise ValueError('Partial stage preserved; register a new attempt instead of overwriting it')
        stage_dir.mkdir(parents=True)
        arguments = effective_arguments(stage, job, settings, workspace)
        if argument(arguments, 'data') != job['dataset'] or int(argument(arguments, 'win_size', 100)) != data_audit['window']:
            raise ValueError('Recipe data/window does not match audit')
        command = [str(ROOT / settings['python']), '-u', 'run_anomaly.py', *arguments]
        before = {str(p) for p in (workspace / 'results').glob('*.csv')}
        stdout, stderr = stage_dir / 'stdout.log', stage_dir / 'stderr.log'
        started = time.time()
        record = {'status': 'running', 'job_id': job['id'], 'stage': stage['id'], 'author_script': stage['script'],
                  'author_arguments': stage['author_arguments'], 'effective_arguments': arguments,
                  'cwd': str(workspace), 'started_unix': started, 'cpu_threads': settings['cpu_threads'],
                  'device': 'cpu', 'stage_fingerprint': fingerprint}
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        with stdout.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
            child = subprocess.Popen(command, cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, creationflags=flags)
        record['pid'] = child.pid
        write_json(stage_receipt, record)
        state.update(active_job=job['id'], active_stage=stage['id'], active_pid=child.pid, updated_unix=time.time())
        write_json(state_path, state)
        code = child.wait()
        record.update(returncode=code, elapsed_seconds=time.time() - started, status='failed' if code else 'validating')
        write_json(stage_receipt, record)
        if code:
            raise RuntimeError(f"Author stage {stage['id']} exited {code}; logs preserved")
        try:
            artifacts, details = stage_artifacts(workspace, before, arguments, data_audit, stdout)
        except Exception as error:
            record.update(status='failed_validation', error=repr(error))
            write_json(stage_receipt, record)
            raise
        record.update(status='completed', artifacts=artifacts, details=details)
        write_json(stage_receipt, record)
        receipts.append(record)
    artifacts = list({r['path']: r for receipt in receipts for r in receipt['artifacts']}.values())
    write_json(final, {'status': 'completed', 'id': job['id'], 'experiment_sha256': fingerprint,
                       'dataset': job['dataset'], 'seed': job['seed'], 'author_recipe': job['author_recipe'],
                       'stages': receipts, 'artifacts': artifacts, 'metrics': receipts[-1]['details']['author_rl_metrics'],
                       'paper_equivalence': False, 'strict_TAB_result': False, 'boundary': settings['boundary']})
    return 'completed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=ROOT / 'configs/experiments/fm_maelnet_author_queue.v1.json')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--job-id')
    args = parser.parse_args()
    queue = read(args.queue)
    settings = queue['settings']
    base = ROOT / settings['output_root']
    lock = exclusive_lock(base / 'cpu_author.lock')
    if lock is None:
        raise RuntimeError('Author lane already has a lock owner')
    files = {r['path']: r['sha256'] for job in queue['jobs'] for r in job['frozen_files']}
    files.update({stage['script']: stage['script_sha256'] for job in queue['jobs'] for stage in job['stages']})
    for path, expected in files.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen author input changed: ' + path)
    versions = read(ROOT / settings['preflight_report'])['environment']['versions']
    probe = 'import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ' + repr(list(versions)) + '}))'
    observed = json.loads(subprocess.check_output([str(ROOT / settings['python']), '-c', probe], text=True))
    if observed != versions:
        raise ValueError('Author dependency environment differs from verified preflight')
    if args.verify_only:
        print(json.dumps({'verified_files': len(files), 'registered_jobs': len(queue['jobs'])}))
        return
    state_path = base / 'status.json'
    state = {'status': 'running', 'pid': os.getpid(), 'queue_sha256': sha(args.queue), 'jobs': {},
             'active_job': None, 'active_stage': None, 'active_pid': None}
    jobs = [job for job in queue['jobs'] if args.job_id is None or job['id'] == args.job_id]
    if not jobs:
        raise ValueError('Unknown requested author job')
    for job in jobs:
        try:
            state['jobs'][job['id']] = run_job(job, settings, state, state_path)
        except Exception as error:
            state['jobs'][job['id']] = 'failed_or_partial_preserved'
            write_json(ROOT / job['output_directory'] / 'failure.json', {'job': job['id'], 'error': repr(error), 'timestamp_unix': time.time()})
        state.update(active_job=None, active_stage=None, active_pid=None, updated_unix=time.time())
        write_json(state_path, state)
    state.update(status='all_attempts_terminal', updated_unix=time.time())
    write_json(state_path, state)


if __name__ == '__main__':
    main()
