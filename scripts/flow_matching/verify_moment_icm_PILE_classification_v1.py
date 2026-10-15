"""Independently close all157 label-free pretraining classification caches.

Inspect actual raw TRAIN cardinalities, stored bytes, shapes, rows and every
coordinate. This does not reimplement normalization/interpolation or train a
model. A cold controller waits for the exact live producer without restarting it.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from scripts.flow_matching import prepare_moment_icm_pile_classification_v1 as producer
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = producer.ROOT, producer.read, producer.sha, producer.write
MODULE = 'scripts.flow_matching.verify_moment_icm_PILE_classification_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all157_classification_closure.2026-10-15_v1.json'


def raw_case_shape(path):
    count = 0; channels = set(); body = False; timestamps = None
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith('#'): continue
            if not body:
                if text.lower().startswith('@timestamps '): timestamps = text.split()[-1].lower()
                if text.lower() == '@data': body = True
            else:
                if timestamps != 'false': raise ValueError('Independent channel count requires original non-timestamp TS format')
                count += 1; channels.add(text.count(':'))
    if not body or count <= 0 or len(channels) != 1 or min(channels) <= 0:
        raise ValueError('Complete nonempty constant-channel raw case stream required')
    return count, channels.pop()


def verify_arrays(directory, record, cases, channels):
    import numpy as np
    directory = Path(directory)
    names = {'values.npy', 'input_mask.npy', 'original_instance_rows.npy'}
    if (set(record['files_sha256']) != names or {p.name for p in directory.glob('*.npy')} != names
            or record['instances'] != cases or record['channels'] != channels
            or record['sequence_length'] != 256 or record['values_shape'] != [cases, channels, 256]
            or record['source_coordinates_copied_and_checked'] != cases * channels * 256
            or record['label_values_exported'] or not record['original_instance_order_and_all_channels_retained']):
        raise ValueError('All original label-free case/channel/time coordinates required')
    for name, expected in record['files_sha256'].items():
        p = directory / name; before = p.stat(); actual = sha(p); after = p.stat()
        if actual != expected or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Stored pretraining array hash changed')
    arrays = []
    try:
        for name in ['values.npy', 'input_mask.npy', 'original_instance_rows.npy']:
            arrays.append(np.load(directory / name, mmap_mode='r', allow_pickle=False))
        values, masks, rows = arrays
        if (values.shape != (cases, channels, 256) or str(values.dtype) != record['values_dtype']
                or masks.shape != (cases, 256) or rows.shape != (cases,)
                or rows.dtype != np.dtype('int64') or not np.array_equal(rows, np.arange(cases, dtype=np.int64))):
            raise ValueError('Complete native shape/dtype and original instance order required')
        # Bounded coordinate scan;the complete saved bytes are never subsampled.
        block = max(1, (16 * 2**20) // max(1, channels * 256 * values.dtype.itemsize))
        for start in range(0, cases, block):
            if not np.isfinite(values[start:start + block]).all():
                raise ValueError('All saved pretraining signal coordinates must be finite')
            if not np.isin(masks[start:start + block], [0, 1]).all():
                raise ValueError('Every saved padding mask coordinate must be binary')
        return dict(instances=cases, channels=channels, coordinates=cases * channels * 256,
            values_dtype=str(values.dtype), mask_dtype=str(masks.dtype), file_sha256=record['files_sha256'],
            all_signal_coordinates_finite=True, all_padding_masks_binary=True,
            complete_raw_case_channel_and_row_reconciliation=True, label_arrays_absent=True)
    finally:
        for array in arrays:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None: mapping.close()


def registration():
    previous = read(ROOT / producer.CONFIG)
    state = ROOT / previous['state_root']
    handle = read(state / 'controller_launch.json')
    if handle['configuration_sha256'] != sha(ROOT / producer.CONFIG): raise ValueError('Exact producer configuration required')
    pins = [producer.CONFIG, 'scripts/flow_matching/prepare_moment_icm_pile_classification_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_classification_v1.py',
        'tests/test_fm_moment_icm_PILE_classification_closure_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=previous['paper_citation'],
        producer_configuration=producer.CONFIG, producer_handle=handle, required_datasets=157,
        required_array_files=471, environment=previous['environment'], python=previous['python'],
        settings=previous['settings'], resource_lock=previous['resource_lock'],
        state_root='experiments/runs/fm_moment_icm_PILE_all157_classification_closure_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all157_classification_closure_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in pins],
        boundary=('Wait for exact existing producer;do not restart source or retry failed numerical/data '
            'attempts. Require all157source proofs/guards and471arrays. Independently re-read rawTRAIN '
            'case/channel counts and SHA256,reconcile full stored shape/dtype/instance order,scan all '
            'signals finite/all masksbinary,validate source padding and rawlengthgate flags. Source '
            'fullTRAINnormalization precedes validation split,not stricttrainonly;no independent scaler/'
            'interpolation-value rewrite or historicalICMpipeline certificate. NoTESTsignal processing, '
            'label arrays,neural fits/inference/SVM. Same nonneural16physical/21commit/16emergency/'
            '3GiBworker resource policy;all existing model/replay guards unchanged. Classification branch '
            'closure is not completePILEpretraining;all other branches and full model runs remain.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen classification closure input changed: ' + pin['path'])
    previous = read(ROOT / config['producer_configuration']); jobs = producer.verify(previous)
    if (config['required_datasets'] != len(jobs) or len(jobs) != 157 or config['required_array_files'] != 471
            or config['settings'] != previous['settings'] or config['environment'] != previous['environment']
            or config['producer_handle']['configuration_sha256'] != sha(ROOT / producer.CONFIG)):
        raise ValueError('Complete all157source scope and original data policy required')
    return previous, jobs


def close_case(config, previous, job):
    name = job['dataset']; source = Path(previous['artifact_root']) / name
    path = source / 'pretraining_preparation.json'
    guard_path = ROOT / previous['state_root'] / 'attempts' / name / 'guard_receipt.json'
    proof, guard = read(path), read(guard_path)
    expected = [previous['python'], '-B', '-X', 'utf8', '-u', '-m', producer.MODULE, '--dataset', name]
    if (not proof['passed'] or proof['configuration_sha256'] != sha(ROOT / producer.CONFIG)
            or proof['job'] != job or proof['environment'] != config['environment']
            or proof['TEST_signal_values_processed'] or proof['model_input_label_values_used']
            or proof['neural_model_fit_inference_SVM'] != 0 or proof['raw_data_or_other_caches_overwritten']
            or proof['downstream_interpolation_exceptions_applied'] or not proof['original_pretraining_pad_policy']
            or proof['strict_train_only_preprocessing'] or proof['historical_ICM_preprocessing_certified']
            or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['proof_sha256'] != sha(path) or guard['exact_child_handle']['command'] != expected):
        raise ValueError('Complete exact source preparation and guard binding required')
    raw = ROOT / job['TRAIN']['path']
    before = raw.stat()
    if sha(raw) != job['TRAIN']['sha256']: raise ValueError('Original TRAIN bytes changed')
    cases, channels = raw_case_shape(raw)
    after = raw.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('Raw TRAIN changed during closure')
    if (cases != proof['complete_raw_TRAIN_audit']['observed_instances']
            or proof['original_raw_length_selected'] != (proof['source_pre_resampling_temporal_length'] >= 256)
            or proof['disclosed_paired_casewise_split'] != producer.split_counts(cases, channels)):
        raise ValueError('Original source gate or disclosed complete-instance split differs')
    record = verify_arrays(source / 'TRAIN', proof['record'], cases, channels)
    return dict(dataset=name, raw_TRAIN_sha256=job['TRAIN']['sha256'], record=record,
        source_proof_sha256=sha(path), source_guard_sha256=sha(guard_path),
        original_raw_length_selected=proof['original_raw_length_selected'],
        source_pre_resampling_temporal_length=proof['source_pre_resampling_temporal_length'],
        disclosed_paired_casewise_split=proof['disclosed_paired_casewise_split'],
        independent_normalization_and_interpolation_reconstructed=False,
        source_fullTRAIN_normalization_not_strict_train_only=True, historical_ICM_pipeline_certified=False)


def worker(config):
    environment = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if environment != config['environment']: raise ValueError('Exact registered data environment required')
    previous, jobs = verify(config); source_state = read(ROOT / previous['state_root'] / 'status.json')
    if source_state['completed_count'] != 157 or source_state['failed_preserved']: raise ValueError('All157actual successful source outcomes required')
    output = Path(config['artifact_root']); output.mkdir(parents=True, exist_ok=False)
    progress = status_writer(output / 'progress.json', write); rows = []
    for job in jobs:
        rows.append(close_case(config, previous, job))
        progress(output / 'progress.json', dict(completed=len(rows), required=157,
            full_signal_coordinates=sum(r['record']['coordinates'] for r in rows)))
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG), environment=environment,
        producer_configuration_sha256=sha(ROOT / producer.CONFIG), datasets=157, array_files=471,
        original_TRAIN_cases=sum(r['record']['instances'] for r in rows),
        full_signal_coordinates=sum(r['record']['coordinates'] for r in rows), records=rows,
        original_raw_length_gate_selected=sum(int(r['original_raw_length_selected']) for r in rows),
        source_normalization_control_preserved=True, strict_train_only_preprocessing=False,
        independent_scaling_and_interpolation_value_reconstruction=False, historical_ICM_pipeline_certified=False,
        full_PILE_loader_integrated=False, neural_training_inference_SVM=0,
        data_overwritten=False, all_original_experiments_complete=False)
    write(output / 'full157_classification_cache_closure.json', report)
    print(json.dumps({k: report[k] for k in ['passed', 'datasets', 'array_files', 'original_TRAIN_cases', 'full_signal_coordinates', 'neural_training_inference_SVM']}))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); previous, _ = verify(config)
    base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing full157closure owner preserved')
    publish = status_writer(base / 'status.json', write)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value=None, **extra):
        state.update(value or {}); state.update(extra); publish(base / 'status.json', state)
    try:
        if (base / 'worker_launch.json').exists(): raise ValueError('Prior closure attempt preserved;no automatic retry')
        while True:
            source = read(ROOT / previous['state_root'] / 'status.json')
            if source['state'] == 'all157_classification_preparation_attempts_terminal':
                if source['completed_count'] != 157 or source['failed_preserved']:
                    update(state='source_failures_preserved_full157closure_not_started', source_completed=source['completed_count'], source_failures=source['failed_preserved'])
                    return
                break
            handle = config['producer_handle']
            try:
                p = psutil.Process(handle['pid'])
                live = p.create_time() == handle['create_time'] and p.cmdline() == handle['command']
            except psutil.NoSuchProcess: live = False
            if not live:
                # Re-read after a natural completion race,without relaunching.
                source = read(ROOT / previous['state_root'] / 'status.json')
                if source['state'] == 'all157_classification_preparation_attempts_terminal': continue
                update(state='producer_handle_missing_without_full157_terminal_proof', source_completed=source['completed_count'])
                return
            update(state='waiting_exact_live_all157_source_producer', source_completed=source['completed_count'], source_pid=p.pid)
            time.sleep(20)
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
            lambda value: update(value, state='waiting_full157closure_data_memory'), poll_seconds=10):
            command = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                write(base / 'worker_launch.json', handle)
                update(state='checking_actual_all157_complete_arrays', active_pid=child.pid)
                code, usage = capped_wait(child, config, api,
                    lambda value: update(value, state='checking_actual_all157_complete_arrays'))
            proof = Path(config['artifact_root']) / 'full157_classification_cache_closure.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
            update(state='full157_independent_cache_closure_passed' if passed else 'failed_preserved_without_retry', active_pid=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen full157closure registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, required=157, array_files=471, numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
