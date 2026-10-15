"""Full157 closure with explicit156original+1recovered provenance and bounded reads.

Every source failure remains preserved. Read-only array mappings are released
after each16MiB block; neither source files nor previous frozen checks change.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from scripts.flow_matching import verify_moment_icm_PILE_classification_v1 as original
from scripts.flow_matching import recover_moment_icm_insect_chunked_export_v2 as recovery
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = original.ROOT, original.read, original.sha, original.write
MODULE = 'scripts.flow_matching.verify_moment_icm_PILE_classification_v2'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all157_classification_closure.2026-10-15_v2.json'


def header(path):
    import numpy as np
    with Path(path).open('rb') as stream:
        if np.lib.format.read_magic(stream) != (1, 0): raise ValueError('Exact original NPY1.0 format required')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        offset = stream.tell()
    if fortran or dtype.hasobject or Path(path).stat().st_size != offset + math.prod(shape) * dtype.itemsize:
        raise ValueError('Complete fixed-size C-order nonobject saved bytes required')
    return shape, dtype, offset


def scan(path, shape, dtype, offset, kind, block_bytes):
    import numpy as np
    row_bytes = math.prod(shape[1:]) * dtype.itemsize
    rows = max(1, block_bytes // row_bytes); blocks = 0; checked = 0; peak = 0
    for start in range(0, shape[0], rows):
        end = min(start + rows, shape[0]); block_shape = (end - start, *shape[1:])
        values = np.memmap(path, mode='r', dtype=dtype, offset=offset + start * row_bytes,
                           shape=block_shape, order='C')
        try:
            if kind == 'signal' and not np.isfinite(values).all(): raise ValueError('All signal coordinates must be finite')
            if kind == 'mask' and not np.isin(values, [0, 1]).all(): raise ValueError('All padding masks must be binary')
            if kind == 'rows' and not np.array_equal(values, np.arange(start, end, dtype=np.int64)):
                raise ValueError('All original instance rows must remain in order')
            blocks += 1; checked += values.size; peak = max(peak, values.nbytes)
        finally:
            values._mmap.close(); del values
    return dict(blocks=blocks, checked_coordinates=checked, maximum_active_mapping_bytes=peak,
                mapping_released_after_every_block=True)


def verify_arrays(directory, record, cases, channels, block_bytes=16 * 2**20):
    import numpy as np
    if isinstance(block_bytes, bool) or not isinstance(block_bytes, int) or block_bytes <= 0:
        raise ValueError('Positive read-mapping block size required')
    directory = Path(directory); names = {'values.npy', 'input_mask.npy', 'original_instance_rows.npy'}
    if (set(record['files_sha256']) != names or {p.name for p in directory.glob('*.npy')} != names
            or record['instances'] != cases or record['channels'] != channels
            or record['sequence_length'] != 256 or record['values_shape'] != [cases, channels, 256]
            or record['source_coordinates_copied_and_checked'] != cases * channels * 256
            or record['label_values_exported'] or not record['original_instance_order_and_all_channels_retained']):
        raise ValueError('Every original label-free case/channel/time coordinate required')
    statistics = {}; dtypes = {}
    for name, shape, kind in [('values.npy', (cases, channels, 256), 'signal'),
                              ('input_mask.npy', (cases, 256), 'mask'),
                              ('original_instance_rows.npy', (cases,), 'rows')]:
        path = directory / name; before = path.stat()
        if sha(path) != record['files_sha256'][name]: raise ValueError('Stored array hash changed')
        actual_shape, dtype, offset = header(path)
        if (actual_shape != shape or (kind == 'signal' and str(dtype) != record['values_dtype'])
                or (kind == 'rows' and dtype != np.dtype('int64'))):
            raise ValueError('Complete native shape/dtype required')
        statistics[kind] = scan(path, shape, dtype, offset, kind, block_bytes)
        dtypes[kind] = str(dtype)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Stored array changed during complete closure')
    if statistics['signal']['checked_coordinates'] != cases * channels * 256:
        raise ValueError('Every signal coordinate must be independently visited')
    return dict(instances=cases, channels=channels, coordinates=cases * channels * 256,
        values_dtype=dtypes['signal'], mask_dtype=dtypes['mask'], file_sha256=record['files_sha256'],
        all_signal_coordinates_finite=True, all_padding_masks_binary=True,
        complete_raw_case_channel_and_row_reconciliation=True, label_arrays_absent=True,
        complete_read_mapping_scans=statistics, whole_signal_array_never_mapped=True)


def validate_recovery_metadata(proof, guard, expected):
    if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['proof_sha256'] != expected['proof_sha256']
            or guard['exact_child_handle']['command'] != expected['worker_command']
            or proof['configuration_sha256'] != expected['configuration_sha256']
            or proof['numerical_source_configuration_sha256'] != expected['numerical_source_configuration_sha256']
            or proof['preserved_failed_guard_sha256'] != expected['preserved_failed_guard_sha256']
            or not proof['diagnosed_chunked_output_mapping_recovery']
            or proof['signal_mapping_chunk_bytes'] != 64 * 2**20
            or proof['original_complete_prefix_cases_compared'] != 5414
            or not proof['original_failed_outcome_not_overwritten']
            or proof['full_original_failed_export_equivalence_certified']):
        raise ValueError('Exact separate recovery command/config/numerical-source/old-failure chain required')


def close_case(config, previous, recovered, job):
    name = job['dataset']; repaired = name == 'InsectWingbeat'
    source = Path(recovered['artifact_root'] if repaired else previous['artifact_root']) / name
    path = source / 'pretraining_preparation.json'; proof = read(path)
    guard_path = (ROOT / recovered['state_root'] / 'guard_receipt.json' if repaired
                  else ROOT / previous['state_root'] / 'attempts' / name / 'guard_receipt.json')
    guard = read(guard_path)
    if repaired:
        expected = dict(proof_sha256=sha(path), configuration_sha256=sha(ROOT / recovery.CONFIG),
            numerical_source_configuration_sha256=sha(ROOT / original.producer.CONFIG),
            preserved_failed_guard_sha256=recovered['previous_failed_guard_sha256'],
            worker_command=[recovered['python'], '-B', '-X', 'utf8', '-u', '-m', recovery.MODULE, '--worker'])
        validate_recovery_metadata(proof, guard, expected)
        for filename, legacy in recovered['preserved_partial_outputs'].items():
            wanted = legacy['complete_prefix_sha256']
            if (recovery.prefix_sha(legacy['path'], legacy['complete_prefix_bytes']) != wanted
                    or recovery.prefix_sha(source / 'TRAIN' / filename, legacy['complete_prefix_bytes']) != wanted
                    or proof['original_partial_prefix_comparisons'][filename] != dict(
                        bytes=legacy['complete_prefix_bytes'], sha256=wanted, byte_equal=True)):
                raise ValueError('Independently repeated preserved old/new complete prefix mismatch')
    else:
        expected = [previous['python'], '-B', '-X', 'utf8', '-u', '-m', original.producer.MODULE, '--dataset', name]
        if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
                or guard['proof_sha256'] != sha(path) or guard['exact_child_handle']['command'] != expected
                or proof['configuration_sha256'] != sha(ROOT / original.producer.CONFIG)):
            raise ValueError('Exact original156 source command/config/guard required')
    if (not proof['passed'] or proof['job'] != job or proof['environment'] != config['environment']
            or proof['TEST_signal_values_processed'] or proof['model_input_label_values_used']
            or proof['neural_model_fit_inference_SVM'] != 0 or proof['raw_data_or_other_caches_overwritten']
            or proof['downstream_interpolation_exceptions_applied'] or not proof['original_pretraining_pad_policy']
            or proof['strict_train_only_preprocessing'] or proof['historical_ICM_preprocessing_certified']):
        raise ValueError('Complete common source transformation policy required')
    raw = ROOT / job['TRAIN']['path']; before = raw.stat()
    if sha(raw) != job['TRAIN']['sha256']: raise ValueError('Original TRAIN bytes changed')
    cases, channels = original.raw_case_shape(raw); after = raw.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('Original TRAIN changed during closure')
    if (cases != proof['complete_raw_TRAIN_audit']['observed_instances']
            or proof['original_raw_length_selected'] != (proof['source_pre_resampling_temporal_length'] >= 256)
            or proof['disclosed_paired_casewise_split'] != original.producer.split_counts(cases, channels)):
        raise ValueError('Original source gate or disclosed paired split changed')
    record = verify_arrays(source / 'TRAIN', proof['record'], cases, channels)
    return dict(dataset=name, source_track='diagnosed_chunked_export_v2' if repaired else 'original_v1',
        source_configuration=recovery.CONFIG if repaired else original.producer.CONFIG,
        source_directory=str(source), raw_TRAIN_sha256=job['TRAIN']['sha256'], record=record,
        source_proof_sha256=sha(path), source_guard_sha256=sha(guard_path),
        original_raw_length_selected=proof['original_raw_length_selected'],
        source_pre_resampling_temporal_length=proof['source_pre_resampling_temporal_length'],
        disclosed_paired_casewise_split=proof['disclosed_paired_casewise_split'],
        independent_normalization_and_interpolation_reconstructed=False,
        source_fullTRAIN_normalization_not_strict_train_only=True, historical_ICM_pipeline_certified=False)


def registration():
    previous = read(ROOT / original.producer.CONFIG); recovered = read(ROOT / recovery.CONFIG)
    state = read(ROOT / previous['state_root'] / 'status.json')
    guard_path = ROOT / recovered['state_root'] / 'guard_receipt.json'; guard = read(guard_path)
    if (state['completed_count'] != 156 or set(state['failed_preserved']) != {'InsectWingbeat'}
            or state['state'] != 'all157_classification_preparation_attempts_terminal'
            or not guard['passed'] or guard['resource_guard_aborted']):
        raise ValueError('All156original successes and actual full recovered success required')
    pins = [original.CONFIG, original.producer.CONFIG, recovery.CONFIG,
        'scripts/flow_matching/verify_moment_icm_PILE_classification_v2.py',
        'scripts/flow_matching/verify_moment_icm_PILE_classification_v1.py',
        'scripts/flow_matching/recover_moment_icm_insect_chunked_export_v2.py',
        'tests/test_fm_moment_icm_PILE_classification_closure_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    return dict(schema_version=2, paper_id='moment_long_context', paper_citation=previous['paper_citation'],
        required_datasets=157, original_source_successes=156, recovered_source_successes=1,
        preserved_original_failed_datasets=['InsectWingbeat'], required_array_files=471,
        producer_configuration=original.producer.CONFIG, recovery_configuration=recovery.CONFIG,
        recovered_guard_sha256=sha(guard_path), read_mapping_block_bytes=16 * 2**20,
        environment=previous['environment'], python=previous['python'], settings=previous['settings'],
        resource_lock=previous['resource_lock'],
        state_root='experiments/runs/fm_moment_icm_PILE_all157_classification_closure_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all157_classification_closure_v2',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in pins],
        boundary=('All157datasets/471arrays with explicit156original+1diagnosedchunkedexport chains, '
            'original failure preserved. Re-read complete rawTRAIN hashes/independentcasechannelcounts; '
            'reconcile full native shape/dtype/rows,allfinite signal/allbinarymask coordinates,all '
            'cached hashes;release readmappings aftereach16MiBblock to avoidwhole10GBresidentmaps. '
            'Independently repeat preserved old/new5414caseprefix hashes. Same data16physical/21commit/'
            '16emergency/3GiBworker policy;original numerical sources and neural guards unchanged. '
            'Source normalization includes futurevalidationcases,not stricttrainonly,no independent '
            'scaler/interpolation rewrite,historicalICMpipeline certificate or actual model training. '
            'Classification branch closure only;remainingPILEbranches/full1epochpretraining/forecast/'
            'SVM/beta/strictcontrols and all other registered experiments remain.'))


def verify(config):
    previous = read(ROOT / original.producer.CONFIG); jobs = original.producer.verify(previous)
    recovered = read(ROOT / recovery.CONFIG); recovery.verify(recovered)
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen bounded closure dependency changed: ' + pin['path'])
    if (config['required_datasets'] != 157 or config['required_array_files'] != 471
            or config['original_source_successes'] != 156 or config['recovered_source_successes'] != 1
            or config['preserved_original_failed_datasets'] != ['InsectWingbeat']
            or config['settings'] != previous['settings'] or config['settings'] != recovered['settings']
            or config['environment'] != previous['environment'] or config['read_mapping_block_bytes'] != 16 * 2**20
            or sha(ROOT / recovered['state_root'] / 'guard_receipt.json') != config['recovered_guard_sha256']):
        raise ValueError('Complete157 separate provenance chains and unchanged full data/resource policy required')
    source = read(ROOT / previous['state_root'] / 'status.json')
    if (set(source['completed_datasets']) != {j['dataset'] for j in jobs} - {'InsectWingbeat'}
            or set(source['failed_preserved']) != {'InsectWingbeat'}):
        raise ValueError('Preserve all156original outcomes and the single failed source track')
    return previous, recovered, jobs


def worker(config):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact source data environment required')
    previous, recovered, jobs = verify(config)
    output = Path(config['artifact_root']); output.mkdir(parents=True, exist_ok=False)
    publish = status_writer(output / 'progress.json', write); rows = []
    for job in jobs:
        rows.append(close_case(config, previous, recovered, job))
        publish(output / 'progress.json', dict(completed=len(rows), required=157,
            full_signal_coordinates=sum(r['record']['coordinates'] for r in rows)))
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), configuration=CONFIG,
        configuration_sha256=sha(ROOT / CONFIG), environment=actual, datasets=157, array_files=471,
        original_source_successes=156, recovered_source_successes=1, original_failed_outcomes_preserved=1,
        original_TRAIN_cases=sum(r['record']['instances'] for r in rows),
        full_signal_coordinates=sum(r['record']['coordinates'] for r in rows), records=rows,
        original_raw_length_gate_selected=sum(int(r['original_raw_length_selected']) for r in rows),
        independent_cached_bytes_rows_shapes_finite_masks_and_raw_counts=True,
        preserved_original_prefix_rechecked=True, bounded_read_mapping_lifetime=True,
        independent_scaling_and_interpolation_value_reconstruction=False, strict_train_only_preprocessing=False,
        historical_ICM_pipeline_certified=False, full_PILE_loader_integrated=False,
        neural_training_inference_SVM=0, data_overwritten=False, all_original_experiments_complete=False)
    write(output / 'full157_classification_cache_closure.json', report)
    print(json.dumps({k: report[k] for k in ['passed', 'datasets', 'array_files', 'original_TRAIN_cases', 'full_signal_coordinates', 'neural_training_inference_SVM']}))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing bounded full157closure owner preserved')
    publish = status_writer(base / 'status.json', write)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value=None, **extra): state.update(value or {}); state.update(extra); publish(base / 'status.json', state)
    try:
        if (base / 'worker_launch.json').exists(): raise ValueError('Prior full closure attempt preserved;no automatic retry')
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
            lambda value: update(value, state='waiting_original_data_memory_policy'), poll_seconds=10):
            command = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time(), command=command)
                write(base / 'worker_launch.json', handle); update(state='checking_all157_complete_bounded_arrays', active_pid=p.pid)
                code, usage = capped_wait(p, config, api, lambda value: update(value, state='checking_all157_complete_bounded_arrays'))
            proof = Path(config['artifact_root']) / 'full157_classification_cache_closure.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
            update(state='full157_independent_bounded_cache_closure_passed' if passed else 'failed_preserved_without_retry', active_pid=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen full157boundedclosure registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, datasets=157, arrays=471, original=156, recovered=1,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
