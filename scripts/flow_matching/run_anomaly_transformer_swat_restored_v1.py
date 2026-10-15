"""Restore the missing SWaT interface around the frozen full author model.

The imported MaelNet CSV loader is an explicitly separate reconstruction track.
It cannot establish the unavailable original SWaT loader or paper row selection.
"""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType

from scripts.flow_matching import run_anomaly_transformer_author_full_v1 as original

ROOT, read, sha, write = original.ROOT, original.read, original.sha, original.write
LOADER_SOURCE = 'experiments/runs/fm_code_completion/sources/maelnet/original/data_provider/data_loader.py'
DATASETS = ['SWaT']


def loader_code(path=None):
    source = (Path(path) if path else ROOT / LOADER_SOURCE).read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='SWATSegLoader']
    if len(nodes) != 1:
        raise ValueError('Exactly one pinned CSV loader class required')
    node = nodes[0]
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]) + '\n'


def patched(relative, source):
    result = original.patched(relative, source)
    if relative == 'solver.py':
        bridge = '''from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
'''+loader_code()+'''
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if dataset != 'SWaT':
        raise ValueError('This explicit restored-loader track is SWaT only')
    selected = SWATSegLoader(root_path=data_path, win_size=win_size, step=1, flag=mode)
    return DataLoader(selected, batch_size=batch_size, shuffle=mode=='train', num_workers=0)
'''
        result = original.replace_once(result,
                'from data_factory.data_loader import get_loader_segment', bridge)
    return result


def verify(config, root=ROOT):
    parent = read(root / config['parent_original_matrix'])
    original.verify(parent, root)
    if sha(root / config['parent_original_matrix']) != config['parent_original_matrix_sha256']:
        raise ValueError('Frozen original full-width matrix changed')
    if config['required_model'] != parent['required_model'] or config['settings'] != parent['settings']:
        raise ValueError('Full author model, budgets and resource protection must be retained')
    if config['original_scope'] != parent['original_scope']:
        raise ValueError('The complete paper/method/data objective must remain')
    if config['seeds'] != parent['seeds'] or len(config['jobs']) != 5:
        raise ValueError('All five original SWaT seed slots required')
    for seed, job in zip(config['seeds'], config['jobs']):
        expected = dict(id=f'anomaly_transformer_author__restored_swat_loader__swat__seed{seed}',
                        dataset='SWaT', seed=seed, epochs=3, anomaly_ratio=0.1,
                        dataset_config='configs/datasets/mtsad_swat.json',
                        status='registered_full_author_runtime')
        if job != expected:
            raise ValueError('Restored loader identity or full author SWaT budget changed')
    if config['loader_source'] != LOADER_SOURCE or config['training_rows_policy'] != 'all_495000_payload_rows_no_trim':
        raise ValueError('Unreviewed source loader or guessed paper row selection')
    if config['loader_restoration_author_equivalence_certified'] is not False:
        raise ValueError('Restoration cannot certify unavailable original loader')
    for pin in config['source_receipts']:
        if sha(root / pin['path']) != pin['sha256']:
            raise ValueError('Frozen restored-loader source changed: '+pin['path'])
    return True


def run_worker(config, job, config_path):
    namespace = dict(vars(original), verify=verify, patched=patched)
    return FunctionType(original.run_worker.__code__, namespace)(config, job, config_path)


def controller(config, config_path):
    from scripts.flow_matching.heavy_resources import guarded_wait
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    import psutil
    verify(config)
    state_root = ROOT / config['state_root']
    lock = exclusive_lock(state_root / 'controller.lock')
    if lock is None:
        raise RuntimeError('The restored SWaT lane already has a live owner')
    common = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                  required_parent_dataset_seed_slots=25, selected_restored_loader_slots=5,
                  original_scope=config['original_scope'], loader_restoration_author_equivalence_certified=False,
                  all_original_experiments_complete=False)
    def update(value):
        write(state_root/'status.json',dict(common,**value))
    for job in config['jobs']:
        base = Path(config['artifact_root'])/job['id']
        receipt = state_root/'receipts'/(job['id']+'.json')
        if (base/'result.json').exists():
            original.check_result(config,job)
            continue
        if base.exists() or receipt.exists():
            update(dict(state='preserved_prior_attempt_requires_review',job=job['id'],active_pid=None))
            continue
        def waiting(value):
            update(dict(value,job=job['id'],active_pid=None))
        with resource_slot(ROOT/config['resource_lock'],config['settings'],waiting,poll_seconds=10):
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                             OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads']))
            with (state_root/(job['id']+'.log')).open('xb') as log:
                process=subprocess.Popen([config['python'],'-B','-X','utf8','-u','-m',
                    'scripts.flow_matching.run_anomaly_transformer_swat_restored_v1',
                    '--config',config_path,'--worker','--job-id',job['id']],cwd=ROOT,env=environment,
                    stdout=log,stderr=subprocess.STDOUT)
                update(dict(state='running_full_width_restored_SWaT_loader',job=job['id'],
                            active_pid=process.pid,active_create_time=psutil.Process(process.pid).create_time()))
                code,peaks=guarded_wait(process,config['settings'],poll_seconds=2)
            passed=code==0 and not peaks['resource_guard_aborted']
            if passed:original.check_result(config,job)
            write(receipt,dict(job=job['id'],exit_code=code,passed=passed,**peaks))
    completed=[]
    for job in config['jobs']:
        receipt=state_root/'receipts'/(job['id']+'.json')
        if receipt.exists() and read(receipt).get('passed'):
            original.check_result(config,job);completed.append(job['id'])
    update(dict(state='selected_restored_loader_attempts_terminal_full_review_still_required',
                active_pid=None,completed_restored_loader_models=completed))
    lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.probe_only:
        import json
        print(json.dumps(dict(passed=True,selected_restored_loader_slots=5,
            required_parent_dataset_seed_slots=25,torch_imported='torch' in sys.modules,
            loader_restoration_author_equivalence_certified=False,all_original_experiments_complete=False)))
    elif args.worker:
        run_worker(config,next(j for j in config['jobs'] if j['id']==args.job_id),ROOT/args.config)
    else:controller(config,args.config)


if __name__=='__main__':
    main()
