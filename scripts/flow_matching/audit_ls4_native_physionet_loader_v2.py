"""Audit actual released LS4 patient loaders at each YAML's original batch size."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait, read, sha, write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, resource_api


def original_batch_budget(yaml, train_count, test_count):
    batch = int(yaml['optim']['batch_size'])
    if yaml['data']['n'] != 8000 or yaml['data']['channel'] != 41 or yaml['data']['classify']:
        raise ValueError('Full original nonclassification PhysioNet scope required')
    return batch, math.ceil(train_count / batch), math.ceil(test_count / batch)


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen native loader source changed: ' + item['path'])
    parser_proof = read(ROOT / config['parser_proof'])
    if parser_proof['total_patients'] != 8000 or parser_proof['variables'] != 41:
        raise ValueError('Full original parser data is required')
    for item in parser_proof['derived_artifacts']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Immutable native patient records changed')
    return parser_proof


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    verify(config)
    lock = read(ROOT / config['environment_lock'])
    if sys.version != lock['python'] or any(importlib.metadata.version(k) != v for k, v in lock['packages'].items()):
        raise ValueError('Original author loader environment changed')
    torch.set_num_threads(config['cpu_threads'])
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve previous full loader audit')
    target.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / config['source_root']))
    import datasets
    import datasets.physionet as native
    def reject_download(*args, **kwargs):
        raise RuntimeError('Complete registered local records required; no implicit raw download')
    native.download_url = reject_download
    with np.load(Path(config['prepared_root']) / 'native_split_and_normalizer.npz', allow_pickle=False) as frozen:
        train_ids = frozen['train_patient_ids'].copy()
        test_ids = frozen['test_patient_ids'].copy()
    rows, profiles = [], []
    for path in config['native_configs']:
        yaml = OmegaConf.load(ROOT / path)
        batch, train_batches, test_batches = original_batch_budget(yaml, len(train_ids), len(test_ids))
        arguments = OmegaConf.create(OmegaConf.to_container(yaml.data, resolve=True))
        arguments.path = config['prepared_root']
        # Call the complete original parse_datasets entry point, not a rebuilt loader.
        objects = datasets.parse_datasets(arguments, batch_size=batch, device=torch.device('cpu'))
        if (objects['input_dim'] != 41 or objects['n_train_batches'] != train_batches
                or objects['n_test_batches'] != test_batches):
            raise ValueError('Released YAML data/batch budget differs')
        for split, expected_ids, expected_batches in (('train', train_ids, train_batches), ('test', test_ids, test_batches)):
            loader = objects[split + '_dataloader']
            actual_ids = np.asarray([record[0] for record in loader.dataset])
            if not np.array_equal(actual_ids, expected_ids) or loader.batch_size != batch:
                raise ValueError('Original patient identity, split order or batch size differs')
            covered, count = 0, 0
            for index, data in enumerate(loader):
                current = len(data['observed_data'])
                if current != min(batch, len(expected_ids) - covered) or data['observed_data'].shape[-1] != 41:
                    raise ValueError('Original full patient batch coverage differs')
                for key in ('observed_data', 'observed_tp', 'observed_mask', 'data_to_predict', 'tp_to_predict', 'mask_predicted_data'):
                    if not torch.isfinite(data[key]).all():
                        raise ValueError('Native full loader output nonfinite')
                rows.append(dict(native_config=path, batch_size=batch, split=split, batch_index=index,
                    patient_ids=expected_ids[covered:covered + current].tolist(),
                    observed_shape=list(data['observed_data'].shape), predicted_shape=list(data['data_to_predict'].shape),
                    observed_count=int(data['observed_mask'].sum()), predicted_count=int(data['mask_predicted_data'].sum())))
                count += 1
                covered += current
                if index % 10 == 0:
                    write(target / 'progress.json', dict(native_config=path, batch_size=batch, split=split,
                        covered_patients=covered, expected_patients=len(expected_ids), phase='original_native_loader'))
            if count != expected_batches or covered != len(expected_ids):
                raise ValueError('All original batches and every patient required')
        profiles.append(dict(native_config=path, batch_size=batch, original_epochs=int(yaml.optim.epochs),
            train_patients=len(train_ids), test_patients=len(test_ids), full_train_batches=train_batches,
            full_test_batches=test_batches, expected_optimizer_updates=int(yaml.optim.epochs) * train_batches,
            extrapolation=bool(arguments.extrap), full_original_loader_passed=True))
        del objects
    verify(config)
    if torch.cuda.is_initialized() or len(rows) != config['expected_full_batches']:
        raise ValueError('Full original CPU loader scope required')
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
        configuration_sha256=sha(ROOT / config_path), full_original_loader_passed=True,
        all_8000_original_patients_and_41_variables=True, original_full_batches=len(rows),
        profiles=profiles, batches=rows, parser_proof=config['parser_proof'],
        earlier_batch64_extrapolation_diagnostic_retained=True, raw_and_prepared_data_untouched=True,
        diagnostic_only=True, performance_result=False, cuda_context_initialized=False,
        all_original_experiments_complete=False, author_equivalence_certified=False, boundary=config['boundary'])
    write(target / 'original_loader_audit.json', proof)
    write(target / 'validation.json', dict(passed=True, source_receipts=config['source_receipts'],
        outputs={'original_loader_audit.json':sha(target / 'original_loader_audit.json')}))


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
        raise RuntimeError('Full native loader audit already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(values):
        if values.get('state') == 'replaying_full_checkpoints':
            values = dict(values, state='auditing_full_original_loader')
        state.update(values)
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [str(ROOT / config['python']), '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.audit_ls4_native_physionet_loader_v2', '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT / config['output_directory'] / 'original_loader_audit.json'
        passed = bool(code == 0 and proof.exists() and read(proof)['full_original_loader_passed'])
        write(base / 'receipt.json', dict(exit_code=code, full_original_loader_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None, full_original_loader_passed=passed))
        if not passed:
            raise RuntimeError('Full original loader failed; preserve all source data and partial proof')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
