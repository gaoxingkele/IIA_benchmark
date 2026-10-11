"""Recheck every full Pi replay stream and independently recalculate its metrics."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import zipfile

import psutil

from scripts.flow_matching.audit_pi_original_recovery_completion_v1 import (
    ROOT, read, digest, binding, count_metrics, interval_pa, same_numbers,
)
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait, write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def verify_terminal(report, receipt, producer, completion):
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
            or not receipt['full_native_checkpoint_reinference_passed']
            or not report['passed'] or not report['full_native_checkpoint_reinference_passed']
            or not report['original_full_checkpoint_loaded_strictly']
            or report['original_id'] != completion['original_id']
            or report['original_result_sha256'] != completion['result_sha256']
            or report['full_parameter_count'] != completion['original_parameters']
            or report['full_native_batch_size'] != 256 or report['cpu_threads'] != producer['cpu_threads']
            or report['completed_original_training_seed_slots'] != 1
            or report['additional_training_seed_slots'] != 0 or receipt['additional_training_seed_slots'] != 0
            or report['strict_TAB_result'] or report['author_equivalence_certified']):
        raise ValueError('The actual complete original native replay must finish without abort or extra seeds')
    streams = report['all_original_score_streams_replayed']
    expected = completion['original_full_score_shapes']
    if set(streams) != set(expected):
        raise ValueError('Every original saved score stream must be present')
    for key, shape in expected.items():
        if (streams[key]['shape'] != shape or not streams[key]['bitwise_equal']
                or streams[key]['maximum_absolute_error'] != 0):
            raise ValueError('The completed original full replay was not bitwise identical: ' + key)
    budget = completion['full_native_budget']
    if (report['rng']['fully_consumed_original_training_epochs'] != budget['epochs_completed']
            or report['rng']['original_train_windows'] != budget['training_windows']
            or report['rng']['original_batch_size'] != 256
            or report['rng']['optimizer_updates_repeated'] != 0
            or report['full_original_training_score_points'] != expected['train_energy'][0]
            or report['full_original_test_score_points'] != completion['scored_test_points']
            or report['dropped_original_test_tail_points'] != completion['dropped_test_tail_points']):
        raise ValueError('All original native sampler draws and scored point coverage must match')


def verify_registration(config):
    for item in config['source_receipts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen completed native replay evidence changed: ' + item['path'])
    producer = read(config['completed_replay_configuration'])
    completion = read(producer['independent_completion'])
    report = read(config['completed_replay_report'])
    receipt = read(producer['state_root'] + '/receipt.json')
    verify_terminal(report, receipt, producer, completion)
    if report['config_sha256'] != digest(config['completed_replay_configuration']):
        raise ValueError('Actual native replay must bind its original frozen configuration')
    case, job, result = binding(producer)
    return producer, completion, report, receipt, case, job, result


def worker(config, config_path):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    producer, completion, report, receipt, case, job, result = verify_registration(config)
    source_path = ROOT / case['job']['output_directory'] / 'checkpoints/author_scores.npz'
    replay_path = ROOT / producer['artifact_directory'] / 'checkpoints/author_scores.npz'
    for path in (source_path, replay_path):
        with zipfile.ZipFile(path) as archive:
            if sum(x.file_size for x in archive.infolist()) > config['maximum_uncompressed_score_bytes']:
                raise ValueError('Complete saved arrays exceed the registered readonly audit budget')
    checked = {}
    with np.load(source_path, allow_pickle=False) as saved, np.load(replay_path, allow_pickle=False) as actual:
        if set(saved.files) != set(actual.files) or set(saved.files) != set(completion['original_full_score_shapes']):
            raise ValueError('Every original and replay stream must be rechecked')
        for key in saved.files:
            original, replayed = saved[key], actual[key]
            if (list(replayed.shape) != completion['original_full_score_shapes'][key]
                    or not np.isfinite(replayed).all() or not np.array_equal(original, replayed)):
                raise ValueError('The full reread score archive differs: ' + key)
            checked[key] = dict(shape=list(replayed.shape), bitwise_equal=True)
            del original, replayed
        labels, energy = actual['labels'], actual['test_energy']
        raw, pa = actual['raw_predictions'], actual['pa_predictions']
        threshold = float(np.percentile(np.concatenate([actual['fused_train'], energy]),
            100 - job['author_config']['data']['anomaly_ratio']))
        if threshold != actual['threshold'] or not np.array_equal(raw, (energy > threshold).astype(int)):
            raise ValueError('Independent original pooled train/test calibration differs')
        if not np.array_equal(interval_pa(labels, raw), pa):
            raise ValueError('Independent original interval point adjustment differs')
        metrics, adjusted_counts = count_metrics(labels, pa)
        pointwise, raw_counts = count_metrics(labels, raw)
        pointwise.update(auroc=float(roc_auc_score(labels, energy)),
                         average_precision=float(average_precision_score(labels, energy)))
        metrics['pointwise_no_PA_same_author_threshold'] = pointwise
        same_numbers(metrics, result['metrics'])
        same_numbers(metrics, completion['independently_recomputed_metrics'])
    verify_registration(config)
    target = ROOT / config['output_directory']
    target.mkdir(parents=True, exist_ok=False)
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        canonical_id=job['id'], seed=job['seed'], dataset=job['dataset'], branch=job['branch'],
        full_original_model_checkpoint_reinference_completed=True,
        all_fifteen_full_saved_and_replay_streams_independently_reread=True,
        independently_rechecked_streams=checked, independently_recomputed_metrics=metrics,
        confusion_no_PA=raw_counts, confusion_author_PA=adjusted_counts, source_threshold=threshold,
        original_score_sha256=digest(source_path), replay_score_sha256=digest(replay_path),
        completed_native_replay_report_sha256=digest(config['completed_replay_report']),
        full_native_budget=completion['full_native_budget'], original_parameters=result['parameter_count'],
        native_replay_resource_receipt=receipt, original_training_seed_slots=1,
        original_required_training_seeds=completion['original_required_training_seeds'],
        additional_training_seed_slots=0, additional_inference_executed_by_this_review=False,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
        source_receipts=config['source_receipts'], configuration=config_path,
        configuration_sha256=digest(config_path), boundary=config['boundary'])
    write(target / 'independent_completed_native_replay.json', proof)
    print(json.dumps(dict(full_original_replay_independently_verified=True, full_streams=len(checked), metrics=metrics)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    verify_registration(config)
    if args.worker:
        worker(config, args.config)
        return
    base = ROOT / config['state_root']
    if base.exists() or (ROOT / config['output_directory']).exists():
        raise FileExistsError('Keep each review attempt immutable')
    base.mkdir(parents=True)
    api = resource_api()
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('The completed-replay review is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, state='starting')
    def update(values):
        state.update(values)
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.verify_pi_original_full_replay_completion_v1', '--config', args.config, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                             PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = psutil.Process(child.pid)
                update(dict(state='reviewing_full_native_replay_arrays', active_pid=child.pid,
                            active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        passed = code == 0 and (ROOT / config['output_directory'] / 'independent_completed_native_replay.json').exists()
        write(base / 'receipt.json', dict(exit_code=code, passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None, passed=passed))
        if not passed:
            raise RuntimeError('Keep the failed full-array review and all source evidence intact')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
