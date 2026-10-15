"""Bind only the two idle frozen controllers' child CUDA mask to explicit -1.

Do not edit pinned models, author ports, jobs, datasets, source budgets or guards.
On this Windows runtime the empty CUDA mask did not reliably disable CUDA.
The original worker's own hardware guard must still prove CUDA unavailable.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import run_anomaly_transformer_author_full_v1 as author
from scripts.flow_matching import run_anomaly_transformer_swat_restored_v1 as restored

ROOT,read,sha,write=author.ROOT,author.read,author.sha,author.write


def verify(config):
    expected={'author':'configs/experiments/fm_anomaly_transformer_author_full.v1.json',
              'restored':'configs/experiments/fm_anomaly_transformer_swat_restored.v1.json'}
    if config['cases']!=expected or config['child_environment_override']!={'CUDA_VISIBLE_DEVICES':'-1'}:
        raise ValueError('Only explicit CUDA invisibility for the two complete frozen matrices is permitted')
    if config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273):
        raise ValueError('Full original project scope required')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen CPU binding source changed: '+pin['path'])
    author.verify(read(ROOT/expected['author']));restored.verify(read(ROOT/expected['restored']))


def bound_popen(command,*,env=None,**kwargs):
    # Construct a new environment: never mutate the caller or another process.
    environment=dict(os.environ if env is None else env)
    environment['CUDA_VISIBLE_DEVICES']='-1'
    return subprocess.Popen(command,env=environment,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--branch',choices=['author','restored'],required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.probe_only:
        import sys
        print(json.dumps(dict(passed=True,branch=args.branch,child_cuda_visible_devices='-1',
            original_native_supported_slots=20,original_SWaT_unavailable_slots=5,restored_SWaT_slots=5,
            torch_imported='torch' in sys.modules,numerical_settings_changed=False)))
        return
    selected=author if args.branch=='author' else restored
    path=config['cases'][args.branch];jobs=read(ROOT/path)
    import psutil
    write(ROOT/config['state_root']/args.branch/'binding.json',dict(pid=os.getpid(),
        create_time=psutil.Process().create_time(),configuration=args.config,configuration_sha256=sha(ROOT/args.config),
        original_configuration=path,original_configuration_sha256=sha(ROOT/path),
        original_controller_module=selected.__name__,child_cuda_visible_devices='-1',
        budgets_models_datasets_guards_unchanged=True,all_original_experiments_complete=False))
    namespace=dict(vars(selected),subprocess=SimpleNamespace(Popen=bound_popen))
    FunctionType(selected.controller.__code__,namespace)(jobs,path)


if __name__=='__main__':main()
