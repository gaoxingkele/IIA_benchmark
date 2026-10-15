"""Prepare every157 source-visible PILE classification TRAIN file for pretraining.

Use source pretraining padding, not downstream classification exceptions. Keep
all cases/channels before a separately disclosed casewise grouped split. No
TEST values, neural fit, corpus reduction or historical equivalence claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from scripts.flow_matching import prepare_moment_icm_classification_v1 as original
from scripts.flow_matching import audit_moment_icm_primary_scope_v1 as raw_audit
from scripts.flow_matching.moment_icm_pretraining_inputs_v1 import ROOT, read, sha, write
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

MODULE = 'scripts.flow_matching.prepare_moment_icm_pile_classification_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all157_classification_pretraining.2026-10-15_v1.json'
SCOPE = 'configs/reproducibility/fm_moment_ICM_pretraining_corpus_audit.2026-10-15_v2.json'


def source_pretraining_dataset(cls, path):
    return cls(seq_len=256, full_file_path_and_name=Path(path).as_posix(), data_split='train',
        scale=True, task_name='pre-training', train_ratio=1., val_ratio=0., test_ratio=0.,
        output_type='univariate', upsampling_pad_direction='backward', upsampling_type='pad',
        downsampling_type='interpolate', pad_mode='constant', pad_constant_values=0,
        random_seed=13)


def export_pretraining_split(dataset, expected_instances, output):
    """Export source signals/masks without requesting labels from pretraining items."""
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    channels = int(dataset.n_channels)
    if len(dataset) != expected_instances * channels:
        raise ValueError('Every original pretraining case and channel required')
    first = dataset[0]
    values = np.lib.format.open_memmap(output / 'values.npy', mode='w+', dtype=first.timeseries.dtype,
        shape=(expected_instances, channels, 256))
    masks = np.lib.format.open_memmap(output / 'input_mask.npy', mode='w+', dtype=first.input_mask.dtype,
        shape=(expected_instances, 256))
    for index in range(len(dataset)):
        item = first if index == 0 else dataset[index]
        case, channel = divmod(index, channels)
        if (item.timeseries.shape != (1, 256) or item.input_mask.shape != (256,)
                or not np.isfinite(item.timeseries).all() or not np.isin(item.input_mask, [0, 1]).all()
                or item.timeseries.dtype != values.dtype or item.input_mask.dtype != masks.dtype):
            raise ValueError('Complete finite original coordinates/dtypes and binary masks required')
        values[case, channel] = item.timeseries[0]
        if channel == 0: masks[case] = item.input_mask
        elif not np.array_equal(masks[case], item.input_mask):
            raise ValueError('All source channels must share original instance padding mask')
        if values[case, channel].tobytes() != item.timeseries[0].tobytes():
            raise ValueError('Saved pretraining source bytes differ')
    values.flush(); masks.flush(); dtype = str(values.dtype)
    del values, masks
    np.save(output / 'original_instance_rows.npy', np.arange(expected_instances, dtype=np.int64), allow_pickle=False)
    return dict(instances=expected_instances, channels=channels, sequence_length=256,
        values_shape=[expected_instances, channels, 256], values_dtype=dtype,
        source_coordinates_copied_and_checked=expected_instances * channels * 256,
        original_instance_order_and_all_channels_retained=True, label_values_exported=False,
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        source_scaler_fit_sample_shape=list(dataset.scaler.mean_.shape) if hasattr(dataset.scaler, 'mean_') else None)


def split_counts(cases, channels):
    for value in [cases, channels]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('All positive complete case/channel counts required')
    ratio = .6 / (.6 + .1)
    grouped = int(ratio * cases)
    source = int(ratio * (cases * channels))
    return dict(grouped_casewise_training_cases=grouped, grouped_casewise_validation_cases=cases - grouped,
        grouped_casewise_training_channel_rows=grouped * channels,
        original_flatten_first_training_channel_rows=source,
        original_flatten_first_boundary_can_split_case=source % channels != 0,
        same_casewise_split_required_for_independent_and_ICM=True,
        exact_historical_ICM_split_certified=False)


def registration():
    scope = read(ROOT / SCOPE)
    previous_path = 'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json'
    previous = read(ROOT / previous_path)
    integrity = read(ROOT / scope['integrity'])
    records = {r['path']: r for r in integrity['records']}
    jobs = []
    for candidate in scope['all_source_getter_paths']['classification']:
        path = Path(candidate); dataset = path.parent.name
        train = (path.parent / (dataset + '_TRAIN.ts')).as_posix()
        if not path.name.endswith('_TEST.ts') or train not in records or candidate not in records:
            raise ValueError('Complete source candidate and original TRAIN pair required')
        jobs.append(dict(dataset=dataset, source_getter_candidate=candidate,
            candidate_integrity=records[candidate], TRAIN=records[train]))
    pins = [SCOPE, previous_path, scope['integrity'],
        'scripts/flow_matching/prepare_moment_icm_pile_classification_v1.py',
        'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
        'scripts/flow_matching/audit_moment_icm_primary_scope_v1.py',
        'tests/test_fm_moment_icm_pile_classification_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    pins += [p['path'] for p in previous['source_receipts'] if '/moment_research/original/' in p['path']]
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=scope['paper_citation'],
        complete_input_scope=SCOPE, source_root=scope['source_root'], environment=previous['environment'],
        python=previous['python'], settings=previous['settings'], resource_lock=previous['resource_lock'],
        jobs=jobs, required_datasets=157, sequence_length=256, scaling='original_full_TRAIN_before_split',
        pretraining_upsampling='pad_all_source_candidates_no_downstream13dataset_interpolation_exceptions',
        train_val_split='disclosed_casewise_prefix6_over7_then_shared_grouping_for_paired_models',
        original_flatten_first_split_retained_as_metadata_only_control=True,
        keep_original_length_eligibility_gate_as_separate_explicit_field=True,
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all157_classification_pretraining_v1',
        state_root='experiments/runs/fm_moment_icm_PILE_all157_classification_pretraining_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(pins))],
        boundary=('All157source-visible classification candidates including raw-short datasets must be '
            'prepared and reported;original raw-length>=256eligibility is explicit,not silently replaced by '
            'post-resampling256. Source class reads TRAIN data and TRAINlabels only although its getter '
            'candidate points to TEST. FullTRAIN raw source transformations/defaultpretraining pad and '
            'interpolate-downsampling retained,no downstream13dataset interpolation override. Full '
            'case/channel/time cache and masks preserve all original training instances;TESTvalues never '
            'processed or merged. The grouped casewise6/7prefix is a disclosed reconstruction shared by '
            'independent/ICM models,not original flatten-first split equivalence. Store original flattened '
            'boundary counts separately. Source constructor internalTRAINlabels never used as model inputs; '
            'signal-only export never requests item labels or writes a labels file. '
            'Related source fullTRAINnormalization includes future validation cases,so not a strict '
            'train-only-normalization certificate. Classification branch only;long/shortforecast/anomaly '
            'branches/fullcorpus selection andactualoneepochtraining remain. Nonneural original data '
            'resource policy16GiBphysical/21GiBcommit/16GiBemergency/3GiBworker unchanged;no model '
            'guard,budget,precision or tolerance changed. Raw/oldcacheddata preserved,no automaticretry.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen PILE classification preparation input changed: ' + pin['path'])
    scope = read(ROOT / config['complete_input_scope'])
    jobs = config['jobs']
    if (config['required_datasets'] != 157 or len(jobs) != 157 or config['sequence_length'] != 256
            or [j['source_getter_candidate'] for j in jobs] != scope['all_source_getter_paths']['classification']
            or len({j['dataset'] for j in jobs}) != 157
            or config['environment'] != original.ENVIRONMENT
            or config['scaling'] != 'original_full_TRAIN_before_split'
            or config['pretraining_upsampling'] != 'pad_all_source_candidates_no_downstream13dataset_interpolation_exceptions'
            or config['train_val_split'] != 'disclosed_casewise_prefix6_over7_then_shared_grouping_for_paired_models'
            or config['settings'] != dict(minimum_available_memory_bytes=16 * 2**30,
                minimum_commit_headroom_bytes=21 * 2**30, emergency_commit_headroom_bytes=16 * 2**30,
                maximum_worker_rss_bytes=3 * 2**30, maximum_worker_private_bytes=3 * 2**30)):
        raise ValueError('Full157source candidates, exact environment and unchanged data protection required')
    for job in jobs:
        if job['TRAIN']['status'] != 'verified' or job['candidate_integrity']['status'] != 'verified':
            raise ValueError('Both raw pair registrations must be verified')
    return jobs


def worker(config, job):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact source parser/producer environment required')
    binding = job['TRAIN']; path = ROOT / binding['path']
    raw = raw_audit.file_audit(path, binding, 'ts')
    if raw['status'] != 'verified': raise ValueError('Original complete TRAIN raw audit failed')
    cls, _, unscaled = original.source_components(ROOT / config['source_root'])
    output = Path(config['artifact_root']) / job['dataset']
    output.mkdir(parents=True, exist_ok=False)
    # Pass the original TEST-shaped getter reference. The source train branch
    # constructs TRAIN internally;it never needs TEST signal values.
    dataset = source_pretraining_dataset(cls, ROOT / job['source_getter_candidate'])
    raw_temporal_length = int(dataset.length_timeseries)
    cases, channel_count = raw['observed_instances'], int(dataset.n_channels)
    if raw['observed_dimensions'] and set(map(int, raw['observed_dimensions'])) != {channel_count}:
        raise ValueError('Every original channel required')
    record = export_pretraining_split(dataset, cases, output / 'TRAIN')
    del dataset; gc.collect()
    if sha(path) != binding['sha256']: raise ValueError('Original TRAIN changed during transformation')
    write(output / 'pretraining_preparation.json', dict(passed=True, configuration_sha256=sha(ROOT / CONFIG),
        job=job, environment=actual, complete_raw_TRAIN_audit=raw, record=record,
        source_pre_resampling_temporal_length=raw_temporal_length,
        original_raw_length_selected=raw_temporal_length >= 256,
        downstream_interpolation_exceptions_applied=False, original_pretraining_pad_policy=True,
        original_data_standardizer_fitting_performed=job['dataset'] not in unscaled,
        source_full_TRAIN_scaler_precedes_validation_split=True, strict_train_only_preprocessing=False,
        disclosed_paired_casewise_split=split_counts(cases, channel_count),
        TEST_signal_values_processed=False, model_input_label_values_used=False,
        historical_ICM_preprocessing_certified=False, neural_model_fit_inference_SVM=0,
        raw_data_or_other_caches_overwritten=False, all_original_experiments_complete=False))


def controller(config, jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing PILE classification owner preserved')
    publish = status_writer(base / 'status.json', write)
    completed = []; failed = {}
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_datasets=157, active_pid=None)
    def update(value=None, **extra):
        state.update(value or {}); state.update(extra)
        state.update(completed_count=len(completed), completed_datasets=list(completed), failed_preserved=dict(failed),
                     captured_utc=datetime.now(timezone.utc).isoformat())
        publish(base / 'status.json', state)
    try:
        for job in jobs:
            name = job['dataset']; attempt = base / 'attempts' / name
            proof = Path(config['artifact_root']) / name / 'pretraining_preparation.json'
            try:
                if attempt.exists() or proof.parent.exists(): raise ValueError('Prior attempt preserved; no automatic retry')
                with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
                    lambda value: update(value, state='waiting_full_PILE_classification_data_memory', active_pid=None, dataset=name), poll_seconds=10):
                    attempt.mkdir(parents=True, exist_ok=False)
                    command = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--dataset', name]
                    with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        handle = dict(pid=process.pid, create_time=psutil.Process(process.pid).create_time(), command=command)
                        write(attempt / 'launch.json', handle)
                        update(state='preparing_actual_complete_PILE_classification_TRAIN', active_pid=process.pid, dataset=name)
                        code, peaks = capped_wait(process, config, api,
                            lambda value: update(value, state='preparing_actual_complete_PILE_classification_TRAIN'))
                    passed = code == 0 and not peaks['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code, **peaks,
                        proof_sha256=sha(proof) if proof.exists() else None, exact_child_handle=handle))
                    if not passed: raise ValueError('Source fullTRAIN preparation failed; preserved without retry')
                completed.append(name)
            except Exception as error:
                failed[name] = dict(type=type(error).__name__, message=str(error))
            update(state='scanning_next_full_PILE_classification_candidate', active_pid=None, dataset=None)
        update(state='all157_classification_preparation_attempts_terminal', active_pid=None, dataset=None)
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--dataset'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Frozen full157registration already exists')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else:
        config = read(ROOT / CONFIG); verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, classification_candidates=157,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.dataset:
        matches = [j for j in config['jobs'] if j['dataset'] == args.dataset]
        if len(matches) != 1: raise ValueError('Exact registered candidate required')
        worker(config, matches[0])
    else: controller(config, config['jobs'])


if __name__ == '__main__': main()
