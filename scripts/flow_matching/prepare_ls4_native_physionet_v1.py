"""Run the released LS4 PhysioNet parser on all original A/B patient files."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait, read, sha, write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, resource_api


def safe_archive_members(archive, target, expected_prefix):
    target = Path(target).resolve()
    files = []
    with tarfile.open(archive, 'r:gz') as source:
        for member in source.getmembers():
            path = (target / member.name).resolve()
            if not path.is_relative_to(target) or member.issym() or member.islnk():
                raise ValueError('Archive contains a path outside the derived raw directory')
            if member.isfile():
                relative = path.relative_to(target)
                if len(relative.parts) != 2 or relative.parts[0] != expected_prefix or path.suffix != '.txt':
                    raise ValueError('Unexpected raw patient archive layout')
                files.append(path.stem)
            elif not member.isdir():
                raise ValueError('Nonregular patient archive member')
    if len(files) != 4000 or len(set(files)) != 4000:
        raise ValueError('All 4000 distinct original patient files required')
    return files


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen native preprocessing source changed: ' + item['path'])


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from sklearn.model_selection import train_test_split
    verify(config)
    lock = read(ROOT / config['environment_lock'])
    if sys.version != lock['python'] or any(importlib.metadata.version(name) != version for name, version in lock['packages'].items()):
        raise ValueError('Frozen native author preprocessing environment differs')
    torch.set_num_threads(config['cpu_threads'])
    if torch.cuda.is_initialized():
        raise ValueError('CPU preprocessing may not initialize CUDA')
    directory = Path(config['prepared_root'])
    report = ROOT / config['output_directory']
    if directory.exists() or report.exists():
        raise FileExistsError('Preserve existing native preprocessing attempt')
    directory.mkdir(parents=True)
    report.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / config['source_root']))
    import datasets.physionet as native
    original_urls = native.PhysioNet.urls.copy()
    # The published URLs include '?download' in the explicit filename, invalid
    # on Windows. Only the URL query and local retrieval route change here.
    native.PhysioNet.urls = [url.split('?', 1)[0] for url in original_urls]
    bindings = {item['url'].split('?', 1)[0]: item for item in config['raw_resources']}
    downloaded = []
    raw_target = directory / 'PhysioNet' / 'raw'
    patient_sets = {}
    for item in config['raw_resources']:
        if item['patient_set'] is not None:
            patient_sets[item['patient_set']] = safe_archive_members(ROOT / item['path'], raw_target, item['patient_set'])
    if set(patient_sets['set-a']) & set(patient_sets['set-b']):
        raise ValueError('Original patient sets overlap')
    def offline_copy(url, raw_folder, filename, md5=None):
        if url not in bindings or Path(raw_folder).resolve() != raw_target.resolve():
            raise ValueError('Unregistered native download target')
        binding = bindings[url]
        source = ROOT / binding['path']
        destination = Path(raw_folder) / filename
        if filename != Path(binding['path']).name or destination.exists() or md5 is not None:
            raise ValueError('Native raw copy must preserve registered canonical filename')
        shutil.copy2(source, destination)
        if sha(destination) != binding['sha256']:
            raise ValueError('Canonical raw-copy checksum changed')
        downloaded.append(dict(path=str(destination), sha256=sha(destination), original_source=binding['path']))
    native.download_url = offline_copy
    write(report / 'progress.json', dict(phase='native_patient_parser', expected_patients=8000))
    dataset_a = native.PhysioNet(str(directory), train=True, download=True,
        quantization=config['quantization'], n_samples=8000, device=torch.device('cpu'))
    dataset_b = native.PhysioNet(str(directory), train=False, download=True,
        quantization=config['quantization'], n_samples=8000, device=torch.device('cpu'))
    records = list(dataset_a.data) + list(dataset_b.data)
    if len(dataset_a) != 4000 or len(dataset_b) != 4000 or len(downloaded) != 3:
        raise ValueError('Full original A/B data and native outcome retrieval required')
    for dataset, expected in ((dataset_a, patient_sets['set-a']), (dataset_b, patient_sets['set-b'])):
        if {r[0] for r in dataset.data} != set(expected):
            raise ValueError('Native parser dropped or duplicated patient identities')
    patient_audit = []
    for patient, times, values, mask, label in records:
        if (values.shape != mask.shape or values.shape != (len(times), 41)
                or not torch.isfinite(times).all() or not torch.isfinite(values).all()
                or not torch.all((mask == 0) | (mask == 1)) or torch.any(times[1:] <= times[:-1])):
            raise ValueError('Invalid full native patient times, masks or feature coverage')
        patient_audit.append(dict(patient_id=patient, time_rows=len(times), observed_values=int(mask.sum()),
                                  first_hour=float(times[0]), last_hour=float(times[-1]), mortality_label_available=label is not None))
    train, test = train_test_split(records, train_size=.8, random_state=42, shuffle=True)
    if len(train) != 6400 or len(test) != 1600 or {r[0] for r in train} & {r[0] for r in test}:
        raise ValueError('Native full patient-grouped 80/20 split differs')
    write(report / 'progress.json', dict(phase='native_full_union_normalizer', parsed_patients=8000))
    data_min, data_max = native.get_data_min_max(records, device='cpu')
    if not torch.isfinite(data_min).all() or not torch.isfinite(data_max).all():
        raise ValueError('Full native feature normalization undefined')
    np.savez_compressed(directory / 'native_split_and_normalizer.npz',
        train_patient_ids=np.asarray([r[0] for r in train]), test_patient_ids=np.asarray([r[0] for r in test]),
        total_patient_ids=np.asarray([r[0] for r in records]), data_min=data_min.numpy(), data_max=data_max.numpy())
    batch_audit = []
    for yaml_path in config['native_configs']:
        arguments = OmegaConf.load(ROOT / yaml_path).data
        for split, members in (('train', train), ('test', test)):
            seen = []
            for index in range(0, len(members), 64):
                patients = members[index:index + 64]
                values = native.variable_time_collate_fn(patients, arguments, torch.device('cpu'),
                    data_type=split, data_min=data_min, data_max=data_max)
                for name in ('observed_data', 'observed_tp', 'data_to_predict', 'tp_to_predict', 'observed_mask', 'mask_predicted_data'):
                    if not torch.isfinite(values[name]).all():
                        raise ValueError('Native full collated data is nonfinite')
                if values['observed_data'].shape[0] != len(patients) or values['observed_data'].shape[-1] != 41:
                    raise ValueError('Native batch dropped patients or variables')
                seen.extend(r[0] for r in patients)
                batch_audit.append(dict(native_config=yaml_path, split=split, batch=index // 64,
                    patient_ids=[r[0] for r in patients], observed_shape=list(values['observed_data'].shape),
                    predicted_shape=list(values['data_to_predict'].shape),
                    observed_count=int(values['observed_mask'].sum()), predicted_count=int(values['mask_predicted_data'].sum()),
                    missing_mortality_labels=int(torch.isnan(values['labels']).sum())))
                if index % (64 * 10) == 0:
                    write(report / 'progress.json', dict(phase='full_native_collate', native_config=yaml_path,
                          split=split, checked_patients=len(seen), expected_patients=len(members)))
            if seen != [r[0] for r in members]:
                raise ValueError('Not every original patient passed the native loader')
    verify(config)
    processed = directory / 'PhysioNet' / 'processed'
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
        configuration_sha256=sha(ROOT / config_path), full_native_preprocessing_passed=True,
        diagnostic_only=True, performance_result=False, total_patients=8000,
        train_patients=6400, test_patients=1600, variables=41, quantization_hours=config['quantization'],
        original_urls=original_urls, offline_windows_safe_urls=native.PhysioNet.urls,
        copied_raw_sources=downloaded, patient_audit=patient_audit, native_batch_audit=batch_audit,
        derived_artifacts=[dict(path=str(p), sha256=sha(p)) for p in sorted(processed.glob('*.pt'))] +
                          [dict(path=str(directory / 'native_split_and_normalizer.npz'), sha256=sha(directory / 'native_split_and_normalizer.npz'))],
        normalization_scope='native_all_8000_patients_including_heldout_test',
        normalization_formula='native_(data-min)/max_not_(max-min); missing positions set to zero',
        interpolation_mask='released_sample_tp_null_cut_tp_null_no_added_heldout_mask',
        patient_order='released_os_listdir_order_frozen_in_full_patient_ids; historic_author_order_unknown',
        outcome_auxiliary_file='native_saved_last_outcome_vector_preserved; per-patient_A_mortality_labels_correct; B_labels_absent; classification_false',
        raw_preserved=True, cuda_context_initialized=torch.cuda.is_initialized(),
        all_original_experiments_complete=False, author_equivalence_certified=False, boundary=config['boundary'])
    write(report / 'native_preprocessing_audit.json', proof)
    write(report / 'validation.json', dict(passed=True, source_receipts=config['source_receipts'],
        outputs={'native_preprocessing_audit.json':sha(report / 'native_preprocessing_audit.json')}))
    print(json.dumps(dict(full_native_patients=8000, native_batches=len(batch_audit), full_native_preprocessing_passed=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify(config)
    if args.worker:
        worker(config, args.config)
        return
    api = resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Native preprocessing already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(values):
        state.update(values)
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [str(ROOT / config['python']), '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.prepare_ls4_native_physionet_v1', '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                             PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                             OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT / config['output_directory'] / 'native_preprocessing_audit.json'
        passed = bool(code == 0 and proof.exists() and read(proof)['full_native_preprocessing_passed'])
        write(base / 'receipt.json', dict(exit_code=code, full_native_preprocessing_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None, full_native_preprocessing_passed=passed))
        if not passed:
            raise RuntimeError('Native full preprocessing failed; preserve original files and partial evidence')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
