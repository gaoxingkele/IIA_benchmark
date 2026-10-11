"""Register the Py37-compatible exact existing full retry without editing live code."""
import copy
from datetime import datetime,timezone
import json
import psutil

from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v3 as entry
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    previous_path='configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v2.json'
    previous=entry.read(entry.ROOT/previous_path)
    native,_,_=entry.verify(previous)
    process=psutil.Process(1540)
    state=entry.read(entry.ROOT/previous['state_root']/'status.json')
    if (state['pid']!=1540 or process.create_time()!=state['create_time']
            or previous_path not in process.cmdline() or state['active_pid'] is not None
            or state['state']!='waiting_for_memory' or entry.previous.actual_native_children(process)
            or (entry.ROOT/native['recovery_job']['output_directory']).exists()):
        raise ValueError('Only the actual lightweight v2 waiting owner may be registered for compatible handoff')
    config=copy.deepcopy(previous)
    config['state_root']='experiments/runs/fm_rcgan_seed1103_registered_retry_dispatch_v3'
    config['native_worker_entry_mode']='direct_full_original_resume'
    config['previous_dispatch_configuration']=previous_path
    config['expected_previous_waiting_owner']=dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline())
    sources={pin['path'] for pin in config['source_receipts']}
    sources.update([previous_path,'scripts/flow_matching/run_rcgan_seed1103_registered_retry_dispatch_v3.py',
        'scripts/flow_matching/register_rcgan_seed1103_registered_retry_dispatch_v3.py',
        'tests/test_fm_rcgan_seed1103_registered_retry_dispatch_v3.py'])
    config['source_receipts']=[dict(path=p,sha256=entry.sha(entry.ROOT/p)) for p in sorted(sources)]
    entry.verify(config)
    path='configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v3.json'
    exclusive_json(entry.ROOT/path,config)
    evidence=entry.ROOT/'projects/flow_matching_research/results/2026-10-12/rcgan_seed1103_registered_retry_dispatch_v3'
    exclusive_json(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=entry.sha(entry.ROOT/path),original_scope=config['original_scope'],
        exact_existing_native_configuration=config['native_configuration'],native_worker_entry_mode=config['native_worker_entry_mode'],
        original_jobs=5,full_epochs_each=500,verified_prior_epochs=238,remaining_full_epochs=262,
        original_scientific_source_functions_unchanged=True,original_guards_unchanged=True,
        original_native_workers_stopped=0,training_started=False,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,old_owner_stopped=False,native_worker_entry_mode=config['native_worker_entry_mode'])))


if __name__=='__main__':main()
