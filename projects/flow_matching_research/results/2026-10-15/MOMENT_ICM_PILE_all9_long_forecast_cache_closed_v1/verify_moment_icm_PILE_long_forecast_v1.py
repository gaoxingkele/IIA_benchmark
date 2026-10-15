"""Close every9x2x3forecast cache cell without pooling source input tracks."""
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

from scripts.flow_matching import prepare_moment_icm_PILE_long_forecast_v1 as producer
from scripts.flow_matching import verify_moment_icm_PILE_classification_v2 as arrays
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = producer.ROOT, producer.read, producer.sha, producer.write
MODULE = 'scripts.flow_matching.verify_moment_icm_PILE_long_forecast_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all9_long_forecast_closure.2026-10-15_v1.json'


def expected_borders(dataset, count):
    if 'ETTm' in dataset: train, val, test = 34560, 11520, 11520
    elif 'ETTh' in dataset: train, val, test = 8640, 2880, 2880
    else: train, test = int(.6 * count), int(.3 * count); val = count - train - test
    raw = dict(train=(0, train), val=(train - 256, train + val), test=(train + val - 256, train + val + test))
    return {key: slice(*value).indices(count)[:2] for key, value in raw.items()}


def overlap(a, b): return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def check_rows(path, start, count):
    import numpy as np
    shape, dtype, offset = arrays.header(path)
    if shape != (count,) or dtype != np.dtype('int64'): raise ValueError('Complete native CSV row indices required')
    step = 16 * 2**20 // 8
    for first in range(0, count, step):
        last = min(first + step, count)
        values = np.memmap(path, mode='r', dtype=dtype, offset=offset + first * 8, shape=(last - first,))
        try:
            if not np.array_equal(values, np.arange(start + first, start + last, dtype=np.int64)):
                raise ValueError('Original nonzero validation/test CSV row origin changed')
        finally: values._mmap.close(); del values


def compare_OT(ot_path, multi_path, feature):
    import numpy as np
    a_shape, a_dtype, a_offset = arrays.header(ot_path)
    b_shape, b_dtype, b_offset = arrays.header(multi_path)
    if a_shape != (b_shape[0], 1) or not 0 <= feature < b_shape[1]: raise ValueError('Aligned separateOT/fullmatrix scope required')
    step = max(1, 16 * 2**20 // (b_shape[1] * b_dtype.itemsize)); changed = 0; maximum = total = 0.
    for start in range(0, b_shape[0], step):
        n = min(step, b_shape[0] - start)
        a = np.memmap(ot_path, mode='r', dtype=a_dtype, offset=a_offset + start * a_dtype.itemsize, shape=(n, 1))
        b = np.memmap(multi_path, mode='r', dtype=b_dtype, offset=b_offset + start * b_shape[1] * b_dtype.itemsize, shape=(n, b_shape[1]))
        try:
            difference = np.abs(a[:, 0] - b[:, feature])
            changed += int(np.count_nonzero(difference)); maximum = max(maximum, float(difference.max(initial=0.)))
            total += float(difference.sum(dtype=np.float64))
        finally: a._mmap.close(); b._mmap.close(); del a, b
    return dict(coordinates=b_shape[0], exactly_different_coordinates=changed, maximum_absolute_difference=maximum,
        mean_absolute_difference=total / b_shape[0], tolerance_applied=False,
        separate_input_tracks_not_pooled=True, this_is_not_model_performance=True)


def close_split(directory, record, expected, expected_channels, train_fit_rows):
    import numpy as np
    directory = Path(directory); start, end = expected; count = end - start
    if (record['time_rows'] != count or record['channels'] != expected_channels
            or record['source_raw_time_start'] != start or record['source_raw_time_end_exclusive'] != end
            or record['complete_source_window_count'] != count - 255 or record['stride'] != 1
            or record['sequence_length'] != 256 or record['source_scaler_fitted_rows'] != train_fit_rows
            or record['source_scaler_n_samples_seen'] != train_fit_rows
            or not record['all_TC_coordinates_saved_byte_equal'] or not record['original_item_transposes_TC_to_CT']
            or record['every_window_independently_replayed'] or record['repeated_window_tensor_materialized']):
        raise ValueError('Complete original split/window/scaler scope required')
    names = {'values': 'values_TC.npy', 'original_rows': 'original_CSV_rows.npy', 'window_mask': 'input_mask_window.npy'}
    if {p.name for p in directory.glob('*.npy')} != set(names.values()): raise ValueError('Exactly3label-free compact arrays percell required')
    statistics = {}; snapshots = {}
    for key, name in names.items():
        path = directory / name; stat = path.stat(); snapshots[name] = (stat.st_size, stat.st_mtime_ns)
        if sha(path) != record['files'][key]['sha256']: raise ValueError('Complete saved forecast array hash changed')
        shape, dtype, offset = arrays.header(path)
        if list(shape) != record['files'][key]['shape'] or str(dtype) != record['files'][key]['dtype']:
            raise ValueError('Native saved array header differs')
        if key == 'values':
            if shape != (count, expected_channels): raise ValueError('Full original time-channel matrix required')
            statistics['signal'] = arrays.scan(path, shape, dtype, offset, 'signal', 16 * 2**20)
        elif key == 'original_rows': check_rows(path, start, count)
        else:
            mask = np.load(path, allow_pickle=False)
            if mask.shape != (256,) or not np.array_equal(mask, np.ones(256)):
                raise ValueError('Original shared fullyobserved source window mask required')
    for name, before in snapshots.items():
        now = (directory / name).stat()
        if before != (now.st_size, now.st_mtime_ns): raise ValueError('Forecast array changed during complete closure')
    for field in ['source_scaler_mean', 'source_scaler_var', 'source_scaler_scale']:
        values = np.asarray(record[field])
        if values.shape != (expected_channels,) or not np.isfinite(values).all(): raise ValueError('Complete finite source scaler vectors required')
    return dict(time_rows=count, channels=expected_channels, coordinates=count * expected_channels,
        source_raw_time_start=start, source_raw_time_end_exclusive=end, window_count=count - 255,
        complete_arrays_rehashed=3, all_signal_coordinates_finite=True, raw_row_origin_and_sequence_verified=True,
        original_constant_window_mask_verified=True, native_headers_and_file_hashes_verified=True,
        independent_scaler_values_recomputed=False, bounded_signal_scan=statistics['signal'])


def registration():
    previous = read(ROOT / producer.CONFIG)
    pins = [producer.CONFIG, 'scripts/flow_matching/prepare_moment_icm_PILE_long_forecast_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_long_forecast_v1.py',
        'tests/test_fm_moment_icm_PILE_long_forecast_closure_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_classification_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=previous['paper_citation'],
        producer_configuration=producer.CONFIG, sources=9, tracks=producer.TRACKS, splits=producer.SPLITS,
        required_cells=54, required_arrays=162, settings=previous['settings'], environment=previous['environment'],
        python=previous['python'], resource_lock=previous['resource_lock'],
        state_root='experiments/runs/fm_moment_icm_PILE_all9_long_forecast_closure_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all9_long_forecast_closure_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in pins],
        boundary=('Close all9x2tracksx3splits/162arrays with independent originalCSVcount/timeborder '
            'reconciliation,all savedhashes/headers/nativeTCcoordinatesfinite,nonzerorawroworigins,'
            'sharedall1maskand source trainingfitrowcount. Record exact input-context overlaps and '
            'ETTunusedrawtail;do not infer targetlabelleakage frominput overlap alone. Sourcecubic '
            'interpolation beforefullCSVsplit remains,strictindependence unproven. Noindependent '
            'source scaler/interpolation numericrewrite. Compare every normalizedOT coordinate '
            'between separatetracks without tolerance/forcingequality;not neuralperformance. '
            'Originaldata16/21/16/3GiBpolicy unchanged;no datamutation/neuralfit/SVM. FullPILE '
            'sampling/remainingbranches/paper1epoch/downstream experiments stillrequired.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen full9closure input changed: ' + pin['path'])
    previous = read(ROOT / config['producer_configuration']); jobs = producer.verify(previous)
    if (config['sources'] != 9 or config['required_cells'] != 54 or config['required_arrays'] != 162
            or config['tracks'] != producer.TRACKS or config['splits'] != producer.SPLITS
            or config['settings'] != previous['settings'] or config['environment'] != previous['environment']):
        raise ValueError('Complete nine-source/two-track/three-split scope and original data protection required')
    return previous, jobs


def worker(config):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact validated data environment required')
    previous, jobs = verify(config); state = read(ROOT / previous['state_root'] / 'status.json')
    if state['completed_count'] != 9 or state['failed_preserved']: raise ValueError('All9actual successful sources required')
    output = Path(config['artifact_root']); output.mkdir(parents=True, exist_ok=False)
    publish = status_writer(output / 'progress.json', write); results = []
    for job in jobs:
        name = job['dataset']; directory = Path(previous['artifact_root']) / name
        proof_path = directory / 'pretraining_preparation.json'; proof = read(proof_path)
        guard_path = ROOT / previous['state_root'] / 'attempts' / name / 'guard_receipt.json'; guard = read(guard_path)
        expected_command = [previous['python'], '-B', '-X', 'utf8', '-u', '-m', producer.MODULE, '--dataset', name]
        if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
                or guard['proof_sha256'] != sha(proof_path) or guard['exact_child_handle']['command'] != expected_command
                or not proof['passed'] or proof['configuration_sha256'] != sha(ROOT / producer.CONFIG)
                or proof['job'] != job or proof['environment'] != actual
                or not proof['originalOT_and_multivariate_tracks_separate'] or not proof['no_TC_CT_axis_patch']
                or not proof['original_source_class_AST_reused'] or proof['neural_training_inference_SVM'] != 0
                or proof['raw_or_other_caches_overwritten'] or proof['historical_ICM_pipeline_certified']):
            raise ValueError('Exact full original preparation and guard chain required')
        raw = producer.raw_audit.file_audit(ROOT / job['raw']['path'], job['raw'], 'csv')
        if raw['status'] != 'verified': raise ValueError('Original complete CSV changed')
        borders = expected_borders(name, raw['observed_instances']); columns = [c for c in raw['CSV_columns'] if c != 'date']
        cells = {}; comparisons = {}
        for track in producer.TRACKS:
            cells[track] = {}
            for split in producer.SPLITS:
                cells[track][split] = close_split(directory / track / split, proof['track_split_records'][track][split],
                    borders[split], 1 if track == 'source_univariate_OT' else len(columns), borders['train'][1])
        for split in producer.SPLITS:
            comparisons[split] = compare_OT(directory / 'source_univariate_OT' / split / 'values_TC.npy',
                directory / 'paired_full_multivariate' / split / 'values_TC.npy', columns.index('OT'))
        results.append(dict(dataset=name, source_proof_sha256=sha(proof_path), source_guard_sha256=sha(guard_path),
            raw_CSV_sha256=job['raw']['sha256'], complete_raw_CSV_rows=raw['observed_instances'],
            independently_reconciled_original_time_borders=borders, raw_tail_outside_original_fixed_splits=raw['observed_instances'] - borders['test'][1],
            input_context_row_overlaps=dict(train_val=overlap(borders['train'], borders['val']),
                val_test=overlap(borders['val'], borders['test']), train_test=overlap(borders['train'], borders['test'])),
            input_overlap_alone_is_not_target_label_leakage_certificate=True,
            independent_target_disjointness_certified=False, cells=cells, separate_normalized_OT_comparisons=comparisons))
        publish(output / 'progress.json', dict(completed_sources=len(results), required_sources=9))
    coordinates = sum(c['coordinates'] for r in results for t in r['cells'].values() for c in t.values())
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), configuration=CONFIG,
        configuration_sha256=sha(ROOT / CONFIG), producer_configuration_sha256=sha(ROOT / producer.CONFIG),
        sources=9, cells=54, array_files=162, full_saved_signal_coordinates=coordinates, records=results,
        environment=actual, all_tracks_splits_independently_bound_and_audited=True,
        original_input_context_overlap_and_unused_ETTtail_retained=True,
        independent_source_scaler_and_interpolation_value_reconstruction=False,
        strict_temporal_independence_certified=False, historical_ICM_pipeline_certified=False,
        complete_PILE_loader_integrated=False, neural_training_inference_SVM=0, data_overwritten=False,
        all_original_experiments_complete=False)
    write(output / 'full9_forecast_cache_closure.json', report)
    print(json.dumps({k: report[k] for k in ['passed', 'sources', 'cells', 'array_files', 'full_saved_signal_coordinates', 'neural_training_inference_SVM']}))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing all9forecast closure owner preserved')
    publish = status_writer(base / 'status.json', write)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value=None, **extra): state.update(value or {}); state.update(extra); publish(base / 'status.json', state)
    try:
        if (base / 'worker_launch.json').exists(): raise ValueError('Prior full9closure attempt preserved;no automaticretry')
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
            lambda value: update(value, state='waiting_original_data_memory_policy'), poll_seconds=10):
            cmd = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time(), command=cmd)
                write(base / 'worker_launch.json', handle); update(state='closing_all9_tracks_splits_and_arrays', active_pid=p.pid)
                code, usage = capped_wait(p, config, api, lambda value: update(value, state='closing_all9_tracks_splits_and_arrays'))
            proof = Path(config['artifact_root']) / 'full9_forecast_cache_closure.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
            update(state='all9_tracks_splits_independent_closure_passed' if passed else 'failed_preserved_without_retry', active_pid=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen full9closure registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only: print(json.dumps(dict(passed=True, sources=9, cells=54, arrays=162, numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
