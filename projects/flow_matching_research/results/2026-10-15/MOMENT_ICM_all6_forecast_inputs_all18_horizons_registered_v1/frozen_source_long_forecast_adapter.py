"""All9 long-forecast sources: separate originalOT and paired multivariate caches.

Reuse the pinned source class/splits/train-fit scaler,keep allTCcoordinates.
Store time matrices rather than expanding overlapping windows256times. The
original pretraining __getitem__ transposes TC to CT correctly;no axis fix.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Optional
import warnings

from scripts.flow_matching import prepare_moment_icm_classification_v1 as components
from scripts.flow_matching import audit_moment_icm_primary_scope_v1 as raw_audit
from scripts.flow_matching.moment_icm_pretraining_inputs_v1 import ROOT, read, sha, write
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

MODULE = 'scripts.flow_matching.prepare_moment_icm_PILE_long_forecast_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all9_long_forecast_pretraining.2026-10-15_v1.json'
SCOPE = 'configs/reproducibility/fm_moment_ICM_pretraining_corpus_audit.2026-10-15_v2.json'
TRACKS = {'source_univariate_OT': 'univariate', 'paired_full_multivariate': 'multivariate'}
SPLITS = ['train', 'val', 'test']


def source_long_class(directory):
    cls, _, _ = components.source_components(directory)
    namespace = dict(cls._read_data.__globals__)
    namespace.update(Optional=Optional, warnings=warnings)
    path = Path(directory) / 'moment/data/forecasting_datasets.py'
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(n, ast.ClassDef) and n.name == 'LongForecastingDataset']
    if len(nodes) != 1: raise ValueError('Complete pinned original LongForecastingDataset required')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['LongForecastingDataset']


def source_dataset(cls, path, track, split):
    if track not in TRACKS or split not in SPLITS: raise ValueError('Explicit source track and time split required')
    return cls(seq_len=256, full_file_path_and_name=Path(path).as_posix(), data_split=split,
        target_col='OT', scale=True, data_stride_len=1, task_name='pre-training',
        train_ratio=.6, val_ratio=.1, test_ratio=.3, output_type=TRACKS[track], random_seed=13)


def save_source_array(path, source, block_bytes=16 * 2**20):
    import numpy as np
    path = Path(path)
    if path.exists(): raise FileExistsError('Preserve existing array output')
    if source.ndim not in (1, 2) or source.shape[0] <= 0 or source.dtype.hasobject:
        raise ValueError('Complete positive numeric vector or time-channel matrix required')
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode='w+', dtype=source.dtype, shape=source.shape, fortran_order=False)
    array.flush(); array._mmap.close(); del array
    rows = max(1, block_bytes // (math.prod(source.shape[1:]) * source.dtype.itemsize))
    blocks = 0
    for start in range(0, source.shape[0], rows):
        end = min(start + rows, source.shape[0])
        array = np.load(path, mmap_mode='r+', allow_pickle=False)
        try:
            array[start:end] = source[start:end]
            if array[start:end].tobytes(order='C') != source[start:end].tobytes(order='C'):
                raise ValueError('Every original stored coordinate must be byte equal')
            if source.dtype.kind == 'f' and not np.isfinite(array[start:end]).all():
                raise ValueError('All source-transformed numeric coordinates must be finite')
            array.flush(); blocks += 1
        finally: array._mmap.close(); del array
    return dict(sha256=sha(path), shape=list(source.shape), dtype=str(source.dtype),
        all_source_coordinates_byte_equal=True, storage_order='C', write_mapping_blocks=blocks,
        mappings_released_after_each_bounded_block=True)


def export_split(dataset, output):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    matrix = dataset.data
    if matrix.ndim != 2 or matrix.shape != (dataset.length_timeseries, dataset.n_channels):
        raise ValueError('Original full time-channel matrix required')
    n = len(dataset)
    if n <= 0 or n != matrix.shape[0] - 256 + 1: raise ValueError('All stride1source windows required')
    borders = dataset._get_borders(); source_slice = getattr(borders, dataset.data_split)
    rows = np.fromiter(range(*source_slice.indices(dataset.length_timeseries_original)), dtype=np.int64)
    if len(rows) != len(matrix): raise ValueError('Complete original CSV time-row mapping required')
    mask = dataset[0].input_mask
    if mask.shape != (256,) or not np.array_equal(mask, np.ones(256)):
        raise ValueError('Original all-observed256timepoint window mask required')
    files = dict(values=save_source_array(output / 'values_TC.npy', matrix),
        original_rows=save_source_array(output / 'original_CSV_rows.npy', rows),
        window_mask=save_source_array(output / 'input_mask_window.npy', mask))
    saved = np.load(output / 'values_TC.npy', mmap_mode='r', allow_pickle=False)
    checked = sorted({0, n // 2, n - 1})
    try:
        for index in checked:
            item = dataset[index]
            if (item.timeseries.shape != (dataset.n_channels, 256)
                    or saved[index:index + 256].T.tobytes() != item.timeseries.tobytes()
                    or item.input_mask.tobytes() != mask.tobytes()):
                raise ValueError('Source window axes, mask or endpoint mapping changed')
    finally: saved._mmap.close(); del saved
    fit_rows = len(range(*borders.train.indices(dataset.length_timeseries_original)))
    return dict(time_rows=len(matrix), channels=dataset.n_channels, source_original_CSV_rows=dataset.length_timeseries_original,
        source_raw_time_start=int(rows[0]), source_raw_time_end_exclusive=int(rows[-1]) + 1,
        stride=1, sequence_length=256, complete_source_window_count=n,
        original_length_gate_selected=dataset.length_timeseries >= 256,
        all_TC_coordinates_saved_byte_equal=True, original_item_transposes_TC_to_CT=True,
        source_window_indices_sampled_for_axis_and_endpoint_checks=checked,
        every_window_independently_replayed=False, repeated_window_tensor_materialized=False,
        scalar_window_mask_shared_by_every_window=True, files=files,
        source_scaler_fitted_rows=fit_rows, source_scaler_n_samples_seen=int(dataset.scaler.n_samples_seen_),
        source_scaler_mean=dataset.scaler.mean_.tolist(), source_scaler_var=dataset.scaler.var_.tolist(),
        source_scaler_scale=dataset.scaler.scale_.tolist(),
        source_scaler_fitted_training_rows_only=True, source_cubic_interpolation_precedes_split=True,
        strict_interpolation_temporal_independence_certified=False,
        validation_and_test_may_include256historical_context_rows=True, actual_neural_model_consumption_verified=False)


def registration():
    scope = read(ROOT / SCOPE)
    previous = read(ROOT / 'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json')
    registry = {r['path']: r for r in read(ROOT / scope['integrity'])['records']}
    jobs = [dict(dataset=Path(p).stem, raw=registry[p]) for p in scope['all_source_getter_paths']['long_forecast']]
    pins = [SCOPE, scope['integrity'],
        'scripts/flow_matching/prepare_moment_icm_PILE_long_forecast_v1.py',
        'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
        'scripts/flow_matching/audit_moment_icm_primary_scope_v1.py',
        'tests/test_fm_moment_icm_PILE_long_forecast_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        scope['source_root'] + '/moment/data/forecasting_datasets.py']
    pins += [p['path'] for p in previous['source_receipts'] if '/moment_research/original/' in p['path']]
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=scope['paper_citation'],
        source_root=scope['source_root'], complete_input_scope=SCOPE, jobs=jobs, required_sources=9,
        tracks=TRACKS, splits=SPLITS, sequence_length=256, data_stride_len=1,
        raw_time_split_policy='original_ETTfixed_months_or_other_sources60_10_30_with256context_overlap',
        scaling='source_train_rows_fit_StandardScaler_then_transform_each_split',
        paired_full_multivariate_is_disclosed_input_scope_control=True,
        original_univariate_OT_track_kept_separate=True,
        environment=previous['environment'], python=previous['python'], settings=previous['settings'],
        resource_lock=previous['resource_lock'],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all9_long_forecast_pretraining_v1',
        state_root='experiments/runs/fm_moment_icm_PILE_all9_long_forecast_pretraining_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(pins))],
        boundary=('All9sources,both originalOT and fullmultivariate tracks,all3originaltime splits;no source '
            'omitted,tracks/splits not pooled. Related MOMENT defaults chooseOTonly;fullmultivariate is '
            'explicit pairedindependent/ICM reconstruction input scope,not silent historical recipe fix. '
            'The original class already returnscorrectTCtransposeCT;no axisbug is alleged or patched. '
            'Keep original trainfitStandardScaler,cubic interpolation overfullCSV beforetime split,and '
            'ETTfixedmonth/other60_10_30borders including256contextoverlap. Do not claim strict temporal '
            'independence or pureTESTunseeninputcontext. AllsourceTCcoordinates copied byteexact/native '
            'dtype to boundedwrite mappings;cachecompact matrices+rawCSVrowids+constant source mask, '
            'not256xexpanded overlappingwindows. First/mid/lastsourcewindows checked,not allwindow '
            'inference replays. No normalizationrewrite,model training/inference/SVM or performance '
            'claim. Original data16physical/21commit/16emergency/3GiBworker policy unchanged;no raw '
            'or oldcache overwritten,no automatic retry. This branch is not completePILEpretraining; '
            'Monash/anomaly/fullsampler/epochfit/nativeproof/downstream/strictcontrols remain.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen all9forecast input changed: ' + pin['path'])
    scope = read(ROOT / config['complete_input_scope'])
    if (config['required_sources'] != 9 or len(config['jobs']) != 9
            or [j['raw']['path'] for j in config['jobs']] != scope['all_source_getter_paths']['long_forecast']
            or config['tracks'] != TRACKS or config['splits'] != SPLITS
            or config['sequence_length'] != 256 or config['data_stride_len'] != 1
            or config['environment'] != components.ENVIRONMENT
            or config['settings'] != dict(minimum_available_memory_bytes=16 * 2**30,
                minimum_commit_headroom_bytes=21 * 2**30, emergency_commit_headroom_bytes=16 * 2**30,
                maximum_worker_rss_bytes=3 * 2**30, maximum_worker_private_bytes=3 * 2**30)):
        raise ValueError('Complete9source/2track/3split scope and unchanged original data policy required')
    return config['jobs']


def worker(config, job):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact producer environment required')
    path = ROOT / job['raw']['path']; raw = raw_audit.file_audit(path, job['raw'], 'csv')
    if raw['status'] != 'verified' or not {'date', 'OT'}.issubset(raw['CSV_columns']):
        raise ValueError('Complete registered original CSV with explicitdate/OTcolumns required')
    cls = source_long_class(ROOT / config['source_root'])
    output = Path(config['artifact_root']) / job['dataset']; output.mkdir(parents=True, exist_ok=False)
    progress = status_writer(output / 'progress.json', write); records = {}
    for track in TRACKS:
        records[track] = {}
        for split in SPLITS:
            dataset = source_dataset(cls, path, track, split)
            if dataset.length_timeseries_original != raw['observed_instances']: raise ValueError('Every original CSV row required')
            expected_channels = 1 if track == 'source_univariate_OT' else len(raw['CSV_columns']) - 1
            if dataset.n_channels != expected_channels: raise ValueError('Exact originalOT/fullmultivariate channel scope required')
            records[track][split] = export_split(dataset, output / track / split)
            del dataset; gc.collect()
            progress(output / 'progress.json', dict(completed_track_split_cells=sum(len(r) for r in records.values()), required=6))
    if sha(path) != job['raw']['sha256']: raise ValueError('Raw CSV changed during preparation')
    write(output / 'pretraining_preparation.json', dict(passed=True, job=job, configuration_sha256=sha(ROOT / CONFIG),
        environment=actual, complete_raw_audit=raw, track_split_records=records,
        original_source_class_AST_reused=True, no_TC_CT_axis_patch=True, originalOT_and_multivariate_tracks_separate=True,
        full_multivariate_scope_is_disclosed_reconstruction=True, strict_interpolation_independence_certified=False,
        historical_ICM_pipeline_certified=False, neural_training_inference_SVM=0,
        raw_or_other_caches_overwritten=False, all_original_experiments_complete=False))


def controller(config, jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing full9forecast source owner preserved')
    publish = status_writer(base / 'status.json', write); completed = []; failed = {}
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, required_sources=9)
    def update(value=None, **extra):
        state.update(value or {}); state.update(extra)
        state.update(completed_count=len(completed), completed_datasets=list(completed), failed_preserved=dict(failed))
        publish(base / 'status.json', state)
    try:
        for job in jobs:
            name = job['dataset']; attempt = base / 'attempts' / name
            proof = Path(config['artifact_root']) / name / 'pretraining_preparation.json'
            try:
                if attempt.exists() or proof.parent.exists(): raise ValueError('Previous source attempt preserved;no automaticretry')
                with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
                    lambda value: update(value, state='waiting_original_data_memory_policy', active_pid=None, dataset=name), poll_seconds=10):
                    attempt.mkdir(parents=True, exist_ok=False)
                    cmd = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--dataset', name]
                    with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        handle = dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time(), command=cmd)
                        write(attempt / 'launch.json', handle); update(state='preparing_all_tracks_original_time_splits', active_pid=p.pid, dataset=name)
                        code, usage = capped_wait(p, config, api, lambda value: update(value, state='preparing_all_tracks_original_time_splits'))
                    passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                        exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
                    if not passed: raise ValueError('Complete original-source preparation failed;preserved withoutretry')
                completed.append(name)
            except Exception as error: failed[name] = dict(type=type(error).__name__, message=str(error))
            update(state='scanning_next_forecast_source', active_pid=None, dataset=None)
        update(state='all9_forecast_preparation_attempts_terminal', active_pid=None, dataset=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--dataset'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen all9forecast registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, sources=9, tracks=2, splits=3,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.dataset:
        matches = [j for j in config['jobs'] if j['dataset'] == args.dataset]
        if len(matches) != 1: raise ValueError('Exact registered forecast source required')
        worker(config, matches[0])
    else: controller(config, config['jobs'])


if __name__ == '__main__': main()
