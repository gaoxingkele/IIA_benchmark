"""Diagnosed RSS recovery for the preserved InsectWingbeat cache attempt.

Only file mapping lifetime changes. The pinned original dataset class, all
25000 cases/200 channels, transformations and original3GiB guards are retained.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching import prepare_moment_icm_pile_classification_v1 as original
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = original.ROOT, original.read, original.sha, original.write
MODULE = 'scripts.flow_matching.recover_moment_icm_insect_chunked_export_v2'
CONFIG = 'configs/reproducibility/fm_moment_ICM_InsectWingbeat_chunked_export.2026-10-15_v2.json'


def chunked_export(dataset, expected_instances, output, max_mapping_bytes=64 * 2**20):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    channels = int(dataset.n_channels)
    if len(dataset) != expected_instances * channels: raise ValueError('All cases and channels required')
    if isinstance(max_mapping_bytes, bool) or not isinstance(max_mapping_bytes, int) or max_mapping_bytes <= 0:
        raise ValueError('Positive output mapping block size required')
    first = dataset[0]; dtype = first.timeseries.dtype; mask_dtype = first.input_mask.dtype
    specifications = [('values.npy', dtype, (expected_instances, channels, 256)),
                      ('input_mask.npy', mask_dtype, (expected_instances, 256))]
    for name, kind, shape in specifications:
        array = np.lib.format.open_memmap(output / name, mode='w+', dtype=kind, shape=shape)
        array.flush(); array._mmap.close(); del array
    rows_per_block = max(1, max_mapping_bytes // (channels * 256 * dtype.itemsize))
    progress = status_writer(output.parent / 'chunked_export_progress.json', write)
    for start in range(0, expected_instances, rows_per_block):
        values = np.load(output / 'values.npy', mmap_mode='r+', allow_pickle=False)
        masks = np.load(output / 'input_mask.npy', mmap_mode='r+', allow_pickle=False)
        try:
            end = min(start + rows_per_block, expected_instances)
            for index in range(start * channels, end * channels):
                item = first if index == 0 else dataset[index]
                case, channel = divmod(index, channels)
                if (item.timeseries.shape != (1, 256) or item.input_mask.shape != (256,)
                        or not np.isfinite(item.timeseries).all() or not np.isin(item.input_mask, [0, 1]).all()
                        or item.timeseries.dtype != dtype or item.input_mask.dtype != mask_dtype):
                    raise ValueError('Complete unchanged source coordinates/dtypes/masks required')
                values[case, channel] = item.timeseries[0]
                if channel == 0: masks[case] = item.input_mask
                elif not np.array_equal(masks[case], item.input_mask): raise ValueError('Source case mask differs between channels')
                if values[case, channel].tobytes() != item.timeseries[0].tobytes():
                    raise ValueError('Saved source coordinates differ')
            values.flush(); masks.flush()
        finally:
            values._mmap.close(); masks._mmap.close(); del values, masks
        progress(output.parent / 'chunked_export_progress.json', dict(
            completed_cases=end, required_cases=expected_instances, channels=channels,
            complete_signal_coordinates=end * channels * 256,
            output_mappings_released_after_block=True))
    np.save(output / 'original_instance_rows.npy', np.arange(expected_instances, dtype=np.int64), allow_pickle=False)
    return dict(instances=expected_instances, channels=channels, sequence_length=256,
        values_shape=[expected_instances, channels, 256], values_dtype=str(dtype),
        source_coordinates_copied_and_checked=expected_instances * channels * 256,
        original_instance_order_and_all_channels_retained=True, label_values_exported=False,
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        source_scaler_fit_sample_shape=list(dataset.scaler.mean_.shape) if hasattr(dataset.scaler, 'mean_') else None)


def prefix_sha(path, byte_count):
    digest = hashlib.sha256(); remaining = byte_count
    with Path(path).open('rb') as stream:
        while remaining:
            data = stream.read(min(2**20, remaining))
            if not data: raise ValueError('Preserved old complete prefix was truncated')
            digest.update(data); remaining -= len(data)
    return digest.hexdigest()


def registration():
    previous = read(ROOT / original.CONFIG)
    name = 'InsectWingbeat'; attempt = ROOT / previous['state_root'] / 'attempts' / name
    guard = read(attempt / 'guard_receipt.json')
    if (guard['passed'] or not guard['resource_guard_aborted'] or guard['proof_sha256'] is not None
            or guard['peak_tree_rss_bytes'] <= previous['settings']['maximum_worker_rss_bytes']
            or guard['peak_tree_private_bytes'] >= previous['settings']['maximum_worker_private_bytes']):
        raise ValueError('Preserved RSS-only capped failed attempt required')
    legacy = Path(previous['artifact_root']) / name / 'TRAIN'
    # v1 exporter writes a mask on the first channel of a case;the last marked
    # case may have incomplete values. Retain only the prior5414 complete cases.
    complete = 5414
    old = {}
    for file, count in [('values.npy', 128 + complete * 200 * 256 * 8),
                        ('input_mask.npy', 128 + complete * 256 * 8)]:
        p = legacy / file; stat = p.stat()
        old[file] = dict(path=str(p), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         complete_prefix_bytes=count, complete_prefix_sha256=prefix_sha(p, count))
    config = deepcopy(previous)
    config.update(previous_configuration=original.CONFIG,
        jobs=[j for j in previous['jobs'] if j['dataset'] == name], recovery_dataset=name,
        previous_failed_guard=str(attempt / 'guard_receipt.json'), previous_failed_guard_sha256=sha(attempt / 'guard_receipt.json'),
        preserved_partial_outputs=old, preserved_complete_prefix_cases=complete,
        recovery='release_output_memmap_every64MiB_without_changing_numerical_transforms',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_InsectWingbeat_chunked_export_v2',
        state_root='experiments/runs/fm_moment_icm_InsectWingbeat_chunked_export_v2')
    pins = {p['path'] for p in previous['source_receipts']}
    pins.update([original.CONFIG, 'scripts/flow_matching/recover_moment_icm_insect_chunked_export_v2.py',
                 'tests/test_fm_moment_icm_chunked_export_v2.py'])
    config['source_receipts'] = [dict(path=p, sha256=sha(ROOT / p)) for p in sorted(pins)]
    config['boundary'] += (' Isolated diagnosed outputmapping recovery for one failed dataset;original all157 '
        'controller and failed outputs remain unchanged. Old partial output entered writer,has5415marked '
        'case masks;conservatively5414complete case prefixes pinned and compared byteexact to recovered '
        'files. Flush/unmap aftereach64MiBsignal block;full class/scaler/resampling/cases/channels/3GiBlimits '
        'unchanged. This is not a blind numerical retry or a historicalICMpipeline certificate.')
    return config


def verify(config):
    previous = read(ROOT / original.CONFIG); original.verify(previous)
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen recovery source changed: ' + pin['path'])
    guard = read(config['previous_failed_guard'])
    if (config['settings'] != previous['settings'] or config['recovery_dataset'] != 'InsectWingbeat'
            or config['jobs'] != [j for j in previous['jobs'] if j['dataset'] == 'InsectWingbeat']
            or config['preserved_complete_prefix_cases'] != 5414 or guard['passed']
            or not guard['resource_guard_aborted'] or sha(config['previous_failed_guard']) != config['previous_failed_guard_sha256']):
        raise ValueError('Original capped failure and unchanged complete numerical/source policy required')
    for record in config['preserved_partial_outputs'].values():
        stat = Path(record['path']).stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Preserve original partial output bytes')


def worker(config):
    prior = original.export_pretraining_split
    try:
        # Fresh worker namespace only. Frozen files and other live interpreters
        # retain their original exporter. All numerical dataset code is reused.
        original.export_pretraining_split = chunked_export
        original.worker(config, config['jobs'][0])
    finally: original.export_pretraining_split = prior
    output = Path(config['artifact_root']) / 'InsectWingbeat'
    proof = read(output / 'pretraining_preparation.json')
    comparisons = {}
    for name, old in config['preserved_partial_outputs'].items():
        before = prefix_sha(old['path'], old['complete_prefix_bytes'])
        new = prefix_sha(output / 'TRAIN' / name, old['complete_prefix_bytes'])
        if before != old['complete_prefix_sha256'] or new != before:
            raise ValueError('Recovered source prefix differs from preserved original numerical output')
        comparisons[name] = dict(bytes=old['complete_prefix_bytes'], sha256=new, byte_equal=True)
    proof.update(configuration_sha256=sha(ROOT / CONFIG), numerical_source_configuration_sha256=sha(ROOT / original.CONFIG),
        diagnosed_chunked_output_mapping_recovery=True, signal_mapping_chunk_bytes=64 * 2**20,
        preserved_failed_guard_sha256=config['previous_failed_guard_sha256'],
        original_complete_prefix_cases_compared=5414, original_partial_prefix_comparisons=comparisons,
        original_failed_outcome_not_overwritten=True, full_original_failed_export_equivalence_certified=False)
    write(output / 'pretraining_preparation.json', proof)


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Recovery already has an owner')
    publish = status_writer(base / 'status.json', write)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value=None, **extra): state.update(value or {}); state.update(extra); publish(base / 'status.json', state)
    try:
        if (base / 'worker_launch.json').exists(): raise ValueError('Preserve prior recovered attempt;no automatic retry')
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'],
            lambda value: update(value, state='waiting_original_data_resource_policy'), poll_seconds=10):
            command = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time(), command=command)
                write(base / 'worker_launch.json', handle); update(state='recovering_full25000cases_200channels', active_pid=p.pid)
                code, usage = capped_wait(p, config, api, lambda value: update(value, state='recovering_full25000cases_200channels'))
            proof = Path(config['artifact_root']) / 'InsectWingbeat' / 'pretraining_preparation.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
            update(state='full_InsectWingbeat_chunked_export_passed' if passed else 'failed_preserved_without_retry', active_pid=None)
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen diagnosed recovery')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else: config = read(ROOT / CONFIG); verify(config)
    if args.probe_only: print(json.dumps(dict(passed=True, cases=25000, channels=200, original_caps_unchanged=True, neural_fits=0)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
