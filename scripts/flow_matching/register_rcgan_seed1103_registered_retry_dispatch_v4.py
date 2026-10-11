"""Freeze the corrected Py37 worker and unambiguous actual --worker invocation."""
import copy
from datetime import datetime,timezone
import json
import psutil
from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v4 as entry
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    prior_path='configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v2.json'
    prior=entry.read(entry.ROOT/prior_path)
    native,_,_=entry.verify(prior)
    process=psutil.Process(1540)
    state=entry.read(entry.ROOT/prior['state_root']/'status.json')
    if (process.create_time()!=state['create_time'] or state['pid']!=1540 or prior_path not in process.cmdline()
            or state['active_pid'] is not None or state['state']!='waiting_for_memory'
            or entry.original.previous.actual_native_children(process)
            or (entry.ROOT/native['recovery_job']['output_directory']).exists()):
        raise ValueError('Actual no-native-child v2 waiting owner required')
    config=copy.deepcopy(prior)
    config.update(state_root='experiments/runs/fm_rcgan_seed1103_registered_retry_dispatch_v4',
        native_worker_entry_mode='direct_full_original_resume',parent_argument_abbreviation_disabled=True,
        previous_dispatch_configuration=prior_path,
        expected_previous_waiting_owner=dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline()))
    sources={pin['path'] for pin in prior['source_receipts']}
    sources.update([prior_path,'configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v3.json',
        'scripts/flow_matching/run_rcgan_seed1103_registered_retry_dispatch_v3.py',
        'scripts/flow_matching/register_rcgan_seed1103_registered_retry_dispatch_v3.py',
        'tests/test_fm_rcgan_seed1103_registered_retry_dispatch_v3.py',
        'scripts/flow_matching/run_rcgan_seed1103_registered_retry_dispatch_v4.py',
        'scripts/flow_matching/register_rcgan_seed1103_registered_retry_dispatch_v4.py',
        'tests/test_fm_rcgan_seed1103_registered_retry_dispatch_v4.py'])
    config['source_receipts']=[dict(path=p,sha256=entry.sha(entry.ROOT/p)) for p in sorted(sources)]
    entry.verify(config)
    path='configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v4.json'
    exclusive_json(entry.ROOT/path,config)
    evidence=entry.ROOT/'projects/flow_matching_research/results/2026-10-12/rcgan_seed1103_registered_retry_dispatch_v4'
    exclusive_json(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=entry.sha(entry.ROOT/path),original_scope=config['original_scope'],
        native_configuration=config['native_configuration'],native_worker_entry_mode=config['native_worker_entry_mode'],
        argument_abbreviation_disabled=True,required_original_jobs=5,original_epochs_each=500,
        verified_prior_epochs=238,remaining_required_epochs=262,old_waiting_owner_stopped=False,
        original_native_workers_stopped=0,additional_training_seed_slots=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,old_waiting_owner_stopped=False)))


if __name__=='__main__':main()
