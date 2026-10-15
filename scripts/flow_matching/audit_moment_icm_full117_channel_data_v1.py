"""Guarded full-coordinate audit of Appendix-D grouped data on all117 caches.

This validates the shared loading component, not full PILE pretraining. Both
official splits are checked separately; TEST is never merged into training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from scripts.flow_matching import moment_icm_channel_data_v1 as channels
from scripts.flow_matching.moment_icm_pretraining_inputs_v1 import ROOT, read, sha, write
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

MODULE = 'scripts.flow_matching.audit_moment_icm_full117_channel_data_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_full117_channel_data.2026-10-15_v1.json'
PREVIOUS = 'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json'
POLICY = 'appendix_D_partition8_remainder_replacement'


def registration():
    previous = read(ROOT / PREVIOUS)
    closure = 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full117_preprocessed_v1/full117_processed_data_closure.json'
    pins = [PREVIOUS, closure, previous['paper_scope'],
        'scripts/flow_matching/moment_icm_channel_data_v1.py',
        'scripts/flow_matching/audit_moment_icm_full117_channel_data_v1.py',
        'tests/test_fm_moment_icm_channel_data_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=previous['paper_citation'],
        previous_preprocessing=PREVIOUS, previous_complete_closure=closure,
        processed_root=previous['artifact_root'], original_preprocessing_state_root=previous['state_root'],
        sequence_length=256, seed=13, channel_policy=POLICY, datasets=117, official_splits=['TRAIN', 'TEST'],
        planning_batch_size_only=2048, environment=previous['environment'], python=previous['python'],
        settings=previous['settings'], resource_lock=previous['resource_lock'],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_full117_channel_data_v1',
        state_root='experiments/runs/fm_moment_icm_full117_channel_data_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in pins],
        reconstruction_choices=dict(channel_order='contiguous_original_indices',
            final_group_oversampling='uniform_with_replacement_from_final_group_only',
            grouping_RNG='NumPyRandomState_uint32_from_first4little_endian_SHA256(seed_NUL_dataset)',
            grouping_fixed_across_cases_and_splits=True, independent_and_ICM_share_same_manifest=True,
            model_RNG_separate_from_batch_reference_RNG=True,
            main_text_subsampling_control_implemented=False),
        boundary=('Audit all936cached file hashes and every grouped coordinate from117prepared classification '
            'datasets/234official splits,not the entire PILE pretraining corpus. Keep original source-control '
            'normalization including TESTfits;not stricttrainonly. Reader never opens labels for model inputs. '
            'Contiguous groups and fixed seeded remainder replacement are disclosed reconstruction choices, '
            'not historical author code certification. Source original channels remain covered even when '
            'final group is oversampled. TRAINandTEST batch plans stay separate. Batch2048 is only a '
            'planning reference from related MOMENT source,not an established ICM paper hyperparameter '
            'or a launched neural budget. No masks/scalers/interpolation/model values rewritten,no neural '
            'fit/inference/SVM. Complete all-branch pretraining data selection,grouped loader integration, '
            'exactsteps/tokens,environment,resources,final checkpoints and native replay remain. '
            'Prior nonneural16GiBphysical/21GiBcommit/16GiBemergency/3GiBworker limits unchanged; '
            'original neural/replay admissions and numerical budgets unchanged.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen grouped-data dependency changed: ' + pin['path'])
    previous = read(ROOT / config['previous_preprocessing'])
    closure = read(ROOT / config['previous_complete_closure'])
    if (config['datasets'] != 117 or config['official_splits'] != ['TRAIN', 'TEST']
            or config['sequence_length'] != 256 or config['seed'] != 13
            or config['channel_policy'] != POLICY or config['planning_batch_size_only'] != 2048
            or config['environment'] != previous['environment'] or config['settings'] != previous['settings']
            or config['processed_root'] != previous['artifact_root']
            or not closure['passed'] or closure['datasets'] != 117 or closure['official_splits'] != 234):
        raise ValueError('Complete117source-control data and explicit frozen channel rules required')
    if [r['dataset'] for r in closure['records']] != [j['dataset'] for j in previous['jobs']]:
        raise ValueError('Every original classification dataset required')
    return closure


class Descriptor:
    def __init__(self, dataset_key, split, width, count):
        self.dataset_key, self.split, self.width, self.count = dataset_key, split, width, count
    def __len__(self): return self.count


def worker(config):
    import numpy as np
    environment = dict(python=platform.python_version(), **{k: importlib.metadata.version(k)
        for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if environment != config['environment']:
        raise ValueError('Exact previously validated producer environment required')
    closure = verify(config)
    output = Path(config['artifact_root'])
    output.mkdir(parents=True, exist_ok=False)
    progress = status_writer(output / 'progress.json', write)
    records = []; descriptors = {'TRAIN': [], 'TEST': []}
    original_coordinates = grouped_coordinates = items = file_count = 0
    for row in closure['records']:
        dataset = row['dataset']
        source = Path(config['processed_root']) / dataset
        guard = ROOT / config['original_preprocessing_state_root'] / 'attempts' / dataset / 'guard_receipt.json'
        if sha(source / 'preprocessing.json') != row['proof_sha256'] or sha(guard) != row['guard_sha256']:
            raise ValueError('Original source/guard binding changed: ' + dataset)
        for split in config['official_splits']:
            expected = row['records'][split]
            directory = source / split
            files = {}
            for name, digest in expected['file_sha256'].items():
                p = directory / name; before = p.stat(); actual = sha(p); after = p.stat()
                if actual != digest or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Previously closed complete cached bytes changed: ' + str(p))
                files[name] = actual; file_count += 1
            data = channels.GroupedCases(directory, dataset_key=dataset, split=split,
                seed=config['seed'], policy=config['channel_policy'])
            try:
                if data.values.shape != (expected['instances'], expected['channels'], 256):
                    raise ValueError('All original case/channel/time coordinates required')
                if str(data.values.dtype) != expected['values_dtype']:
                    raise ValueError('Source dtype changed')
                groups = data.groups
                counts = np.bincount([c for group in groups for c in group], minlength=data.channels)
                if len(counts) != data.channels or (counts < 1).any():
                    raise ValueError('No original channel may be omitted')
                digest = hashlib.sha256(); case_visits = np.zeros(data.cases, dtype=np.int64)
                for index in range(len(data)):
                    item = data.item(index); case = item['case_index']; case_visits[case] += 1
                    if item['original_instance_row'] != case or item['split'] != split:
                        raise ValueError('Original official instance identity changed')
                    if item['input_mask'].tobytes() != data.mask[case].tobytes():
                        raise ValueError('Padding mask changed')
                    # Independent per-slot source indexing verifies every emitted
                    # coordinate,including all oversampled duplicates.
                    for slot, original_channel in enumerate(item['channel_indices']):
                        if item['values'][slot].tobytes() != data.values[case, original_channel].tobytes():
                            raise ValueError('Emitted grouped signal differs from original source coordinate')
                    digest.update(json.dumps([dataset, split, case, item['group_index'], item['channel_indices']],
                        separators=(',', ':')).encode('utf-8'))
                    digest.update(item['values'].tobytes()); digest.update(item['input_mask'].tobytes())
                if not (case_visits == len(groups)).all():
                    raise ValueError('Every case must retain every channel group')
                emitted = len(data) * data.width * 256
                original_coordinates += expected['coordinates']; grouped_coordinates += emitted; items += len(data)
                records.append(dict(dataset=dataset, split=split, original_instances=data.cases,
                    original_channels=data.channels, original_coordinates=expected['coordinates'],
                    grouped_items=len(data), group_width=data.width, channel_groups=groups,
                    source_channel_multiplicity=counts.tolist(), grouped_coordinates=emitted,
                    every_coordinate_and_mask_byte_equal_to_bound_source=True,
                    every_original_case_and_channel_retained=True, native_dtype=str(data.values.dtype),
                    grouped_input_stream_sha256=digest.hexdigest(), cached_files_sha256=files,
                    label_file_read_as_model_input=False))
                descriptors[split].append(Descriptor(dataset, split, data.width, len(data)))
            finally:
                data.close()
            progress(output / 'progress.json', dict(phase='checking_every_real_grouped_coordinate',
                completed_splits=len(records), original_coordinates=original_coordinates,
                grouped_coordinates=grouped_coordinates, grouped_items=items))
    plans = {}
    for split, data in descriptors.items():
        batches = channels.paired_epoch_batches(data, config['planning_batch_size_only'], config['seed'])
        seen = [np.zeros(len(d), dtype=np.uint8) for d in data]
        digest = hashlib.sha256(); widths = {}
        for batch in batches:
            widths[batch.width] = widths.get(batch.width, 0) + 1
            for source_index, item in batch.items:
                if data[source_index].width != batch.width or seen[source_index][item] != 0:
                    raise ValueError('Duplicated/mismatched paired epoch reference')
                seen[source_index][item] += 1
            digest.update(json.dumps([batch.width, batch.items], separators=(',', ':')).encode('utf-8'))
        if not all((s == 1).all() for s in seen):
            raise ValueError('Paired shape buckets lost items or a final partial batch')
        plans[split] = dict(planning_reference_batch_size=2048, batch_count=len(batches),
            batches_by_channel_width=widths, dataset_order=[d.dataset_key for d in data],
            grouped_items=sum(len(d) for d in data), every_reference_exactly_once=True,
            batch_reference_stream_sha256=digest.hexdigest(), same_manifest_required_for_both_models=True,
            actual_model_consumption_verified=False, full_PILE_training_epoch=False)
    if file_count != 936 or len(records) != 234 or original_coordinates != 267291904:
        raise ValueError('Full previously closed117case/channel coverage required')
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG), environment=environment,
        datasets=117, official_splits=234, complete_cached_files_rehashed=file_count,
        original_signal_coordinates=original_coordinates, grouped_signal_coordinates=grouped_coordinates,
        grouped_case_items=items, records=records, separate_split_planning_manifests=plans,
        channel_policy=config['channel_policy'], reconstruction_choices=config['reconstruction_choices'],
        all_original_channels_retained=True, label_values_not_loaded_as_model_inputs=True,
        source_control_normalization_preserved=True, strict_train_only_preprocessing=False,
        historical_ICM_pipeline_certified=False, full_PILE_training_loader_integrated=False,
        neural_fit_inference_SVM=0, raw_or_processed_data_overwritten=False,
        all_original_experiments_complete=False)
    write(output / 'full117_grouped_coordinate_audit.json', report)
    print(json.dumps(dict(passed=True, datasets=117, splits=234, files=936,
        original_coordinates=original_coordinates, grouped_coordinates=grouped_coordinates,
        grouped_items=items, neural_fits=0)))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base / 'controller.lock')
    if lane is None: raise RuntimeError('Existing grouped-data audit owner preserved')
    publish = status_writer(base / 'status.json', write)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value): state.update(value); publish(base / 'status.json', state)
    try:
        if (base / 'worker_launch.json').exists(): raise ValueError('Preserve the previous attempt; no automatic retry')
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update, poll_seconds=10):
            command = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--config', CONFIG, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=process.pid, create_time=psutil.Process(process.pid).create_time(), command=command)
                write(base / 'worker_launch.json', handle)
                update(dict(state='checking_all117_actual_grouped_data', active_pid=process.pid, active_create_time=handle['create_time']))
                code, usage = capped_wait(process, config, api, lambda value: update(dict(value, state='checking_all117_actual_grouped_data')))
            proof = Path(config['artifact_root']) / 'full117_grouped_coordinate_audit.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, proof_sha256=sha(proof) if proof.exists() else None))
            update(dict(state='all117_grouped_coordinate_audit_passed' if passed else 'failed_preserved_without_retry', active_pid=None))
    finally:
        lane.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()
    if args.config != CONFIG: raise ValueError('Frozen exact audit configuration required')
    if args.register:
        if (ROOT / CONFIG).exists(): raise FileExistsError('Preserve frozen channel-data registration')
        config = registration(); verify(config)
        with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
    else:
        config = read(ROOT / CONFIG); verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, datasets=117, splits=234,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
