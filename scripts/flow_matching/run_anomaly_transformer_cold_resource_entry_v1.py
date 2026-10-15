"""Cold resource bindings for four exact, idle Anomaly Transformer controllers.

Expose only the frozen resource functions through lightweight import facades.
Numerical children start independent interpreters and receive original commands.
"""
import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT,read,sha,write

ENTRIES={
 'author':('scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v3',
           ['--config','configs/runtime/fm_anomaly_transformer_explicit_cpu_binding.v3.json','--branch','author'],
           'configs/experiments/fm_anomaly_transformer_author_full.v1.json',25),
 'restored':('scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v3',
           ['--config','configs/runtime/fm_anomaly_transformer_explicit_cpu_binding.v3.json','--branch','restored'],
           'configs/experiments/fm_anomaly_transformer_swat_restored.v1.json',5),
 'paper':('scripts.flow_matching.run_anomaly_transformer_paper_contract_v1',
           ['--config','configs/experiments/fm_anomaly_transformer_paper_contract.v1.json'],
           'configs/experiments/fm_anomaly_transformer_paper_contract.v1.json',25),
 'sensitivity':('scripts.flow_matching.run_anomaly_transformer_paper_sensitivity_v1',
           ['--config','configs/experiments/fm_anomaly_transformer_paper_sensitivity.v1.json'],
           'configs/experiments/fm_anomaly_transformer_paper_sensitivity.v1.json',400)}


def install_facades(first_snapshot=None):
    names=['scripts.flow_matching.heavy_resources','scripts.flow_matching.heavy_scheduler_v2',
           'scripts.flow_matching.run_tsad_queue']
    if any(name in sys.modules for name in names):
        raise RuntimeError('Install resource facades only in a fresh cold metadata interpreter')
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(first_snapshot or (lambda _:None))
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    api=resource_api()
    exports={names[0]:['resource_snapshot','has_headroom','terminate_tree','guarded_wait'],
             names[1]:['resource_slot'],names[2]:['exclusive_lock']}
    for name,functions in exports.items():
        module=ModuleType(name)
        module.__dict__.update({key:api[key] for key in functions})
        module.__dict__['cold_frozen_resource_facade']=True
        sys.modules[name]=module
    return api


def verify(config,role):
    if config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273):
        raise ValueError('Keep the entire original objective')
    if set(config['roles'])!=set(ENTRIES):raise ValueError('All four exact lanes required')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen cold entry dependency changed: '+pin['path'])
    selected=config['roles'][role];module,arguments,path,count=ENTRIES[role]
    if (selected['entry_module']!=module or selected['arguments']!=arguments or
        selected['queue']!=path or len(read(ROOT/path)['jobs'])!=count):
        raise ValueError('Exact original entry, arguments and complete matrix required')
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--role',choices=ENTRIES,required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);selected=verify(config,args.role)
    observed=[False]
    def first_snapshot(snapshot):
        if observed[0]:return
        import os,psutil
        from datetime import datetime,timezone
        process=psutil.Process()
        write(ROOT/config['state_root']/args.role/'actual_first_cached_snapshot.json',dict(
            captured_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),create_time=process.create_time(),
            resource_snapshot=snapshot,private_bytes=process.memory_info().private,
            numpy_imported='numpy' in sys.modules,pandas_imported='pandas' in sys.modules,
            torch_imported='torch' in sys.modules,entry_module=selected['entry_module'],
            original_arguments=selected['arguments'],complete_original_jobs=ENTRIES[args.role][3],
            numerical_workers_started_by_this_observer=0,original_thresholds_changed=False))
        observed[0]=True
    api=install_facades(None if args.probe_only else first_snapshot)
    if args.probe_only:
        # Import the exact original entry and resolve its guard API, without
        # running its main, acquiring its live lane lock, or starting a model.
        importlib.import_module(selected['entry_module'])
        import psutil
        print(json.dumps(dict(passed=True,role=args.role,complete_original_jobs=ENTRIES[args.role][3],
            snapshot=api['resource_snapshot'](),snapshot_function=api['resource_snapshot'].__name__,
            private_bytes=psutil.Process().memory_info().private,
            numpy_imported='numpy' in sys.modules,pandas_imported='pandas' in sys.modules,
            torch_imported='torch' in sys.modules,numerical_worker_executed=False)))
        return
    sys.argv=[selected['entry_module'],*selected['arguments']]
    runpy.run_module(selected['entry_module'],run_name='__main__',alter_sys=True)


if __name__=='__main__':main()
