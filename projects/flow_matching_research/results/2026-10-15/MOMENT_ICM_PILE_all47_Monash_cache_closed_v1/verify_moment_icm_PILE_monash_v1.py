"""All47 source-normalized Monash cache closure with independent outline checks.

Visit every saved signal/mask coordinate and reconcile raw names/lengths,
special chunks, raw-order identities,seed13permutation and all3split positions.
No independent normalization/interpolation rewrite or model performance claim.
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

from scripts.flow_matching import prepare_moment_icm_PILE_monash_v1 as producer
from scripts.flow_matching import verify_moment_icm_PILE_classification_v2 as arrays
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = producer.ROOT, producer.read, producer.sha, producer.write
MODULE = 'scripts.flow_matching.verify_moment_icm_PILE_monash_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all47_Monash_closure.2026-10-15_v1.json'


def outlines(path, special_names):
    columns = []; body = False; raw_index = 0; special = Path(path).stem in special_names
    with Path(path).open('r', encoding='cp1252') as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith('#'): continue
            if not body:
                if text.startswith('@attribute'):
                    parts = text.split(' '); columns.append((parts[1], parts[2]))
                elif text.startswith('@data'): body = True
                continue
            fields = text.split(':')
            if len(fields) != len(columns) + 1: raise ValueError('Raw metadata cardinality mismatch')
            names = [n for n, _ in columns]; index = names.index('series_name')
            name = int(fields[index]) if columns[index][1] == 'numeric' else fields[index]
            length = fields[-1].count(',') + 1
            if special:
                if raw_index != 0 or len(columns) != 2: raise ValueError('One original special longseries required')
                for chunk in range(length // 256):
                    yield dict(series_name='T' + str(chunk + 1), pre_resampling_length=256,
                        raw_series_index=0, source_chunk_index=chunk, raw_time_start=chunk * 256,
                        raw_time_end_exclusive=(chunk + 1) * 256, original_special_tail_omitted=length % 256)
            else:
                yield dict(series_name=name, pre_resampling_length=length, raw_series_index=raw_index,
                    source_chunk_index=None, raw_time_start=0, raw_time_end_exclusive=length, original_special_tail_omitted=0)
            raw_index += 1
    if not body or raw_index == 0: raise ValueError('Complete original TSF rows required')


def verify_cache(directory, record, profile, raw_path, special_names):
    import numpy as np
    directory = Path(directory); count = profile['source_length_field']
    names = {'values.npy', 'input_mask.npy', 'source_series_rows.npy', 'source_shuffle_order.npy',
             'source_series_lengths.npy', 'series_metadata.jsonl'}
    if (set(record['files_sha256']) != names or {p.name for p in directory.glob('*.npy')} != names - {'series_metadata.jsonl'}
            or record['series'] != count or record['channels'] != 1 or record['sequence_length'] != 256
            or not record['source_order_cache_with_exact_shuffle_index']):
        raise ValueError('All original label-free Monash source items and indices required')
    snapshots = {}
    for name in names:
        path = directory / name; stat = path.stat(); snapshots[name] = (stat.st_size, stat.st_mtime_ns)
        if sha(path) != record['files_sha256'][name]: raise ValueError('Complete cached Monash bytes changed')
    vectors = {}
    for name in ['source_series_rows.npy', 'source_shuffle_order.npy', 'source_series_lengths.npy']:
        vector = np.load(directory / name, allow_pickle=False)
        if vector.shape != (count,) or vector.dtype != np.dtype('int64'): raise ValueError('Complete native index vectors required')
        vectors[name] = vector
    if not np.array_equal(vectors['source_series_rows.npy'], np.arange(count, dtype=np.int64)):
        raise ValueError('Raw source row order changed')
    expected_order = np.random.RandomState(13).choice(count, size=count, replace=False)
    if not np.array_equal(vectors['source_shuffle_order.npy'], expected_order):
        raise ValueError('Original full seed13shuffle permutation differs')
    lengths = vectors['source_series_lengths.npy']; observed = 0
    with (directory / 'series_metadata.jsonl').open('r', encoding='utf-8') as cached:
        for index, expected in enumerate(outlines(raw_path, special_names)):
            if index >= count: raise ValueError('Raw row count exceeds source profile')
            line = cached.readline()
            if not line: raise ValueError('Missing source row metadata')
            actual = json.loads(line)
            if actual != dict(source_row=index, **expected) or lengths[index] != expected['pre_resampling_length']:
                raise ValueError('Independent raw name/length/chunk/row provenance mismatch')
            observed += 1
        if cached.read().strip(): raise ValueError('Extra source metadata rows')
    if observed != count or (lengths <= 0).any(): raise ValueError('Every original positive series length required')
    n_train, n_test = int(.6 * count), int(.3 * count)
    expected_splits = dict(train=[0, n_train], val=[n_train, count - n_test], test=[count - n_test, count])
    if record['original_split_positions'] != expected_splits: raise ValueError('Original complete split offsets differ')
    train = lengths[expected_order[:n_train]]
    stats = dict(actual_training_rows=n_train, training_temporal_min=int(train.min()) if n_train else None,
        training_temporal_max=int(train.max()) if n_train else None, training_temporal_total_points=int(train.sum()),
        training_rows_length_ge_window=int((train >= 256).sum()))
    if record['observed_source_training_length_metadata'] != stats or any(profile[k] != v for k, v in stats.items()):
        raise ValueError('Original per-training-series length metadata differs')
    if (record['source_count_gate_selected'] != (count >= 256)
            or record['separate_temporal_control_selected'] != bool(n_train and train.max() >= 256)):
        raise ValueError('Keep original count-field and separate temporal eligibility explicit')
    vshape, vdtype, voffset = arrays.header(directory / 'values.npy')
    mshape, mdtype, moffset = arrays.header(directory / 'input_mask.npy')
    if (vshape != (count, 1, 256) or mshape != (count, 256)
            or str(vdtype) != record['dtype'] or str(mdtype) != record['mask_dtype']):
        raise ValueError('Full native signal/mask header required')
    block = max(1, 16 * 2**20 // (256 * vdtype.itemsize)); coordinates = padded = blocks = 0
    for start in range(0, count, block):
        stop = min(start + block, count); rows = stop - start
        values = np.memmap(directory / 'values.npy', mode='r', dtype=vdtype,
            offset=voffset + start * 256 * vdtype.itemsize, shape=(rows, 1, 256))
        masks = np.memmap(directory / 'input_mask.npy', mode='r', dtype=mdtype,
            offset=moffset + start * 256 * mdtype.itemsize, shape=(rows, 256))
        try:
            wanted = np.arange(256)[None, :] >= np.maximum(0, 256 - lengths[start:stop, None])
            if not np.isfinite(values).all() or not np.array_equal(masks, wanted):
                raise ValueError('All source signal coordinates finite and exact source padding-mask layout required')
            if np.any(values[:, 0, :][~wanted] != 0): raise ValueError('Original backward-pad constant0 differs')
            coordinates += values.size; padded += int((~wanted).sum()); blocks += 1
        finally: values._mmap.close(); masks._mmap.close(); del values, masks
    for name, before in snapshots.items():
        stat = (directory / name).stat()
        if before != (stat.st_size, stat.st_mtime_ns): raise ValueError('Monash cache changed during complete closure')
    return dict(series=count, signal_coordinates=coordinates, mask_coordinates=count * 256,
        source_padding_coordinates=padded, read_mapping_blocks=blocks, arrays=5, metadata_rows=count,
        all_raw_names_lengths_chunks_and_indices_reconciled=True, exact_seed13_shuffle_and_splits=True,
        all_signals_finite_and_exact_source_padding_masks=True, raw_order_source_identity_preserved=True,
        original_count_gate_selected=count >= 256, separate_temporal_control_selected=bool(n_train and train.max() >= 256),
        source_training_rows=n_train, original_split_positions=expected_splits,
        complete_normalized_resampling_values_independently_reconstructed=False)


def registration():
    import ast
    previous = read(ROOT / producer.CONFIG)
    path = ROOT / previous['source_root'] / 'moment/data/forecasting_datasets.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ShortForecastingDataset')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_list_of_long_timeseries')
    special = ast.literal_eval(next(n for n in fn.body if isinstance(n, ast.Return)).value)
    pins = [producer.CONFIG, 'scripts/flow_matching/prepare_moment_icm_PILE_monash_v1.py',
        'scripts/flow_matching/moment_icm_monash_streaming_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_monash_v1.py',
        'tests/test_fm_moment_icm_PILE_monash_closure_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_classification_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=previous['paper_citation'],
        producer_configuration=producer.CONFIG, sources=47, required_arrays=235, special_source_names=special,
        producer_handle=read(ROOT / previous['state_root'] / 'controller_launch.json'),
        environment=previous['environment'], python=previous['python'], settings=previous['settings'],
        resource_lock=previous['resource_lock'],
        state_root='experiments/runs/fm_moment_icm_PILE_all47_Monash_closure_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all47_Monash_closure_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in pins],
        boundary=('All47sources/235arraysand47metadataJSONL/rawhashes;independent rawname/length/chunk '
            'reconciliation,fullsourceorderIDs/seed13permutation/splitoffsets,everyfinitecoordinate/'
            'exactlengthbasedbackwardpadmask/constant0padding. Countgate33vsseparatetemporalgate28 '
            'andtrainitembudgets371162vs278699muststayexplicit. No independent fullnormalized '
            'resampling value rewrite;nativebyteequal testscover6real sources/allitems/3splits,not '
            'entire historical authorframe for largefiles. Source normalization fullseries before '
            'masking retained,not strictmaskedtargetindependent. Originalnonneural16/21/16/3GiB '
            'policy unchanged;raw/cachesuntouched,no neuralfit/SVM,allotherPILE/modeltasksremain.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen Monash closure input changed: ' + pin['path'])
    previous = read(ROOT / config['producer_configuration']); jobs = producer.verify(previous)
    if (config['sources'] != 47 or config['required_arrays'] != 235
            or config['environment'] != previous['environment'] or config['settings'] != previous['settings']
            or len(config['special_source_names']) != 5
            or config['producer_handle']['configuration_sha256'] != sha(ROOT / producer.CONFIG)):
        raise ValueError('Complete47scope and original resource/environment required')
    return previous, jobs


def worker(config):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact validated source data environment required')
    previous, jobs = verify(config); state = read(ROOT / previous['state_root'] / 'status.json')
    if state['completed_count'] != 47 or state['failed_preserved']: raise ValueError('All47actual successful source preparations required')
    output = Path(config['artifact_root']); output.mkdir(parents=True, exist_ok=False)
    publish = status_writer(output / 'progress.json', write); results = []
    for job in jobs:
        name = job['dataset']; directory = Path(previous['artifact_root']) / name
        path = directory / 'pretraining_preparation.json'; proof = read(path)
        receipt = ROOT / previous['state_root'] / 'attempts' / name / 'guard_receipt.json'; guard = read(receipt)
        expected = [previous['python'], '-B', '-X', 'utf8', '-u', '-m', producer.MODULE, '--dataset', name]
        if (not proof['passed'] or proof['configuration_sha256'] != sha(ROOT / producer.CONFIG)
                or proof['job'] != job or proof['environment'] != actual
                or not proof['source_AST_normalization_and_getitem_reused'] or not proof['scaleTrue_source_values_materialized']
                or proof['neural_training_inference_SVM'] != 0 or proof['raw_and_other_caches_overwritten']
                or proof['original_historical_ICM_pipeline_certified'] or not guard['passed']
                or guard['exit_code'] != 0 or guard['resource_guard_aborted']
                or guard['proof_sha256'] != sha(path) or guard['exact_child_handle']['command'] != expected):
            raise ValueError('Full original normalized source/guard/command binding required')
        raw = ROOT / job['raw']['path']; stat = raw.stat()
        if sha(raw) != job['raw']['sha256']: raise ValueError('Original rawTSF changed')
        result = verify_cache(directory, proof['record'], job['metadata_profile'], raw, config['special_source_names'])
        now = raw.stat()
        if (stat.st_size, stat.st_mtime_ns) != (now.st_size, now.st_mtime_ns): raise ValueError('RawTSF changed during independentoutline audit')
        results.append(dict(dataset=name, source_proof_sha256=sha(path), source_guard_sha256=sha(receipt),
            raw_TSF_sha256=job['raw']['sha256'], cache_directory=str(directory), result=result))
        publish(output / 'progress.json', dict(completed_sources=len(results), required_sources=47))
    selected = [r for r in results if r['result']['original_count_gate_selected']]
    temporal = [r for r in results if r['result']['separate_temporal_control_selected']]
    if len(selected) != 33 or len(temporal) != 28: raise ValueError('Both complete metadata controls required')
    original_items = sum(r['result']['source_training_rows'] for r in selected)
    temporal_items = sum(r['result']['source_training_rows'] for r in temporal)
    if original_items != 371162 or temporal_items != 278699: raise ValueError('Original fulltraining item budgets differ')
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), configuration=CONFIG,
        configuration_sha256=sha(ROOT / CONFIG), producer_configuration_sha256=sha(ROOT / producer.CONFIG),
        sources=47, arrays=235, metadata_JSONL_files=47, records=results, environment=actual,
        total_source_series=sum(r['result']['series'] for r in results),
        full_signal_coordinates=sum(r['result']['signal_coordinates'] for r in results),
        original_count_gate_sources=33, separate_temporal_control_sources=28,
        original_selected_training_items=original_items, temporal_control_training_items=temporal_items,
        all_cached_coordinates_finite_exact_masks_and_raw_metadata_reconciled=True,
        independent_full_normalized_resampling_values_reconstructed=False,
        strict_masked_target_independent_normalization=False, historical_ICM_pipeline_certified=False,
        full_PILE_loader_integrated=False, neural_fit_inference_SVM=0, data_overwritten=False,
        all_original_experiments_complete=False)
    write(output / 'full47_Monash_cache_closure.json', report)
    print(json.dumps({k: report[k] for k in ['passed', 'sources', 'arrays', 'total_source_series', 'full_signal_coordinates', 'neural_fit_inference_SVM']}))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Existing full47closure owner preserved')
    publish = status_writer(base / 'status.json', write)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value=None, **extra): state.update(value or {}); state.update(extra); publish(base / 'status.json', state)
    try:
        if (base / 'worker_launch.json').exists(): raise ValueError('Prior closure attempt preserved;no automaticretry')
        previous, _ = verify(config)
        while True:
            source = read(ROOT / previous['state_root'] / 'status.json')
            if source['state'] == 'all47_Monash_preparation_attempts_terminal':
                if source['completed_count'] != 47 or source['failed_preserved']:
                    update(state='source_failures_preserved_complete_closure_not_started',
                        source_completed=source['completed_count'], source_failures=source['failed_preserved'])
                    return
                break
            handle = config['producer_handle']
            try:
                p = psutil.Process(handle['pid'])
                live = p.create_time() == handle['create_time'] and p.cmdline() == handle['command']
            except psutil.NoSuchProcess: live = False
            if not live:
                source = read(ROOT / previous['state_root'] / 'status.json')
                if source['state'] == 'all47_Monash_preparation_attempts_terminal': continue
                update(state='producer_handle_missing_without_full47_terminal_proof', source_completed=source['completed_count'])
                return
            update(state='waiting_exact_live_all47_normalized_source_preparation', source_pid=p.pid,
                source_completed=source['completed_count'])
            time.sleep(20)
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
            lambda value: update(value, state='waiting_original_data_memory_policy'), poll_seconds=10):
            cmd = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time(), command=cmd)
                write(base / 'worker_launch.json', handle); update(state='checking_all47_complete_Monash_caches', active_pid=p.pid)
                code, usage = capped_wait(p, config, api, lambda value: update(value, state='checking_all47_complete_Monash_caches'))
            proof = Path(config['artifact_root']) / 'full47_Monash_cache_closure.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
            update(state='all47_independent_Monash_cache_closure_passed' if passed else 'failed_preserved_without_retry', active_pid=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen full47closure registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only: print(json.dumps(dict(passed=True, sources=47, arrays=235, numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
