"""Prepare all47 Monash sources with source-exact normalized streaming items.

Keep every raw-order source row plus exact seed13shuffle/split references.
Original count-field eligibility and temporal-length control stay separate.
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

from scripts.flow_matching import moment_icm_monash_streaming_v1 as streaming
from scripts.flow_matching import prepare_moment_icm_classification_v1 as components
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = streaming.ROOT, streaming.read, streaming.sha, streaming.write
MODULE = 'scripts.flow_matching.prepare_moment_icm_PILE_monash_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all47_Monash_pretraining.2026-10-15_v1.json'
SCOPE = 'configs/reproducibility/fm_moment_ICM_pretraining_corpus_audit.2026-10-15_v2.json'
PROFILE = 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_pretraining_scope_recovery_v2/actual_full_corpus_and_temporal_gate_audit.json'


def registration():
    scope = read(ROOT / SCOPE); profile = read(ROOT / PROFILE)
    registry = {r['path']: r for r in read(ROOT / scope['integrity'])['records']}
    profiles = {r['path']: r for r in profile['Monash_profiles']}
    previous_path = 'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json'
    previous = read(ROOT / previous_path)
    jobs = [dict(dataset=Path(path).stem, raw=registry[path], metadata_profile=profiles[path])
            for path in scope['all_source_getter_paths']['short_forecast']]
    pins = [SCOPE, PROFILE, previous_path, scope['integrity'],
        'scripts/flow_matching/prepare_moment_icm_PILE_monash_v1.py',
        'scripts/flow_matching/moment_icm_monash_streaming_v1.py',
        'scripts/flow_matching/moment_icm_pretraining_inputs_v1.py',
        'scripts/flow_matching/moment_icm_streaming_scope_v2.py',
        'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
        'tests/test_fm_moment_icm_monash_streaming_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        scope['source_root'] + '/moment/data/forecasting_datasets.py',
        scope['source_root'] + '/configs/default.yaml', scope['source_root'] + '/configs/pretraining/pretrain.yaml']
    pins += [p['path'] for p in previous['source_receipts'] if '/moment_research/original/' in p['path']]
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=scope['paper_citation'],
        source_root=scope['source_root'], input_scope=SCOPE, metadata_profile=PROFILE, jobs=jobs,
        required_sources=47, sequence_length=256, source_seed=13,
        source_ratios=dict(train=.6, validation=.1, test=.3), normalization='exact_source_per_series_mean_std_ddof0_plus1e-7',
        upsampling='source_backward_pad', downsampling='source_pretraining_YAML_interpolate',
        class_default_last_crop_is_not_pretraining_YAML_policy=True,
        cache_order='raw_order_source_items_plus_exact_source_shuffle_and_split_indices',
        original_count_field_gate_selected_sources=33, separate_temporal_control_selected_sources=28,
        original_gate_selected_training_items=371162, temporal_control_selected_training_items=278699,
        settings=previous['settings'], environment=previous['environment'], python=previous['python'],
        resource_lock=previous['resource_lock'],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all47_Monash_pretraining_v1',
        state_root='experiments/runs/fm_moment_icm_PILE_all47_Monash_pretraining_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(pins))],
        boundary=('All47registeredTSFs retained,including datasets excludedbyeither gate. Raw snapshot '
            'has0missing/nonfinite tokens;rejectunexpectedtokens ratherthan quietlycoercing author '
            'stringNaN handling. Stream one rawseries atatime;Pythonfloat tokensandfloat64Series.array '
            'matchoriginalconverter;special5singlelongseries splitintofloorL/256chunksbefore '
            'normalization andomitexactsource tail. PreservePandasArrayordinary/NumPyArrayspecial '
            'normalization types,executeoriginal ASTlambda andoriginal getitem with exactrowfields/'
            'metadata/options. YAMLdownsampleinterpolate differsfromclassdefaultlast,explicitlyfixed '
            'toactualpretrainingYAML. Allsourceorderitemcoords/masksbytechecked;sourceDataFramesample '
            'seed13indexand6/.1/.3splitreferences retained. Noeagerallrowframe,no modelrandommask/'
            'neuralfit/inference/SVM;no historicalICMpipelinecertificate. Source normalizesfull '
            'series beforemodelmasking/resampling,notstricttrain-only/maskedtargetindependentcontrol. '
            'Original sourcegate usesseriescount>=256;separatetemporalcontrolnotpooled orsilently '
            'substituted. Same nonneural16physical/21commit/16emergency/3GiBworkerpolicy;alloriginal '
            'model/replayguardsunchanged,no raw/oldcacheoverwrite,noautomaticretry. Fullindependent '
            'cacheaudit,anomalybranch,entirePILEsampler/steps/token/full1epochmodelrunsremain.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen full47Monash source changed: ' + pin['path'])
    scope = read(ROOT / config['input_scope']); profiles = read(ROOT / config['metadata_profile'])
    if (config['required_sources'] != 47 or len(config['jobs']) != 47
            or [j['raw']['path'] for j in config['jobs']] != scope['all_source_getter_paths']['short_forecast']
            or config['sequence_length'] != 256 or config['source_seed'] != 13
            or config['source_ratios'] != dict(train=.6, validation=.1, test=.3)
            or config['normalization'] != 'exact_source_per_series_mean_std_ddof0_plus1e-7'
            or config['downsampling'] != 'source_pretraining_YAML_interpolate'
            or config['environment'] != components.ENVIRONMENT
            or config['settings'] != dict(minimum_available_memory_bytes=16 * 2**30,
                minimum_commit_headroom_bytes=21 * 2**30, emergency_commit_headroom_bytes=16 * 2**30,
                maximum_worker_rss_bytes=3 * 2**30, maximum_worker_private_bytes=3 * 2**30)):
        raise ValueError('Complete47originalseries/options/environment and original resource policy required')
    for job, profile in zip(config['jobs'], profiles['Monash_profiles']):
        if job['metadata_profile'] != profile or profile['raw_missing_tokens'] or profile['raw_nonfinite_numeric_tokens']:
            raise ValueError('Every verified complete raw profile and exact missing-value scope required')
    return config['jobs']


def worker(config, job):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact native source parser/numerical environment required')
    path = ROOT / job['raw']['path']; before = path.stat()
    if sha(path) != job['raw']['sha256']: raise ValueError('Registered rawTSF changed')
    cls = streaming.sources.short_forecast_class(ROOT / config['source_root'])
    normalize = streaming.source_normalizer(ROOT / config['source_root'])
    output = Path(config['artifact_root']) / job['dataset']
    record = streaming.prepare(path, cls, normalize, job['metadata_profile'], output)
    after = path.stat()
    if sha(path) != job['raw']['sha256'] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Original rawTSF changed during complete preparation')
    write(output / 'pretraining_preparation.json', dict(passed=True, configuration_sha256=sha(ROOT / CONFIG),
        job=job, environment=actual, record=record, source_AST_normalization_and_getitem_reused=True,
        scaleTrue_source_values_materialized=True, raw_and_other_caches_overwritten=False,
        source_count_vs_temporal_data_controls_separate=True, normalization_is_not_strict_mask_independent=True,
        original_historical_ICM_pipeline_certified=False, neural_training_inference_SVM=0,
        all_original_experiments_complete=False))


def controller(config, jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing all47Monash owner preserved')
    publish = status_writer(base / 'status.json', write); completed = []; failed = {}
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, required_sources=47)
    def update(value=None, **extra):
        state.update(value or {}); state.update(extra)
        state.update(completed_count=len(completed), completed_datasets=list(completed), failed_preserved=dict(failed))
        publish(base / 'status.json', state)
    try:
        for job in jobs:
            name = job['dataset']; attempt = base / 'attempts' / name
            proof = Path(config['artifact_root']) / name / 'pretraining_preparation.json'
            try:
                if attempt.exists() or proof.parent.exists(): raise ValueError('Prior source attempt preserved;no automaticretry')
                with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
                    lambda value: update(value, state='waiting_original_data_memory_policy', active_pid=None, dataset=name), poll_seconds=10):
                    attempt.mkdir(parents=True, exist_ok=False)
                    cmd = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--dataset', name]
                    with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        handle = dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time(), command=cmd)
                        write(attempt / 'launch.json', handle); update(state='normalizing_and_resampling_all_source_series', active_pid=p.pid, dataset=name)
                        code, usage = capped_wait(p, config, api, lambda value: update(value, state='normalizing_and_resampling_all_source_series'))
                    passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                        exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
                    if not passed: raise ValueError('Complete source normalization/resampling failed;preserved withoutretry')
                completed.append(name)
            except Exception as error: failed[name] = dict(type=type(error).__name__, message=str(error))
            update(state='scanning_next_Monash_source', active_pid=None, dataset=None)
        update(state='all47_Monash_preparation_attempts_terminal', active_pid=None, dataset=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--dataset'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen full47Monash registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only: print(json.dumps(dict(passed=True, sources=47, normalized_source_items=True,
        numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.dataset:
        matches = [j for j in config['jobs'] if j['dataset'] == args.dataset]
        if len(matches) != 1: raise ValueError('Exact registered Monash source required')
        worker(config, matches[0])
    else: controller(config, config['jobs'])


if __name__ == '__main__': main()
