"""Cold handoff of registered idle full-scope Spectral/CRNNGAN/image monitors.

Execute the same frozen entry modules and arguments. Only freshly compiled host
resource_snapshot bindings reuse the existing cached Windows telemetry types.
Every original numerical child, queue, dataset, budget and guard remains intact.
"""
import argparse
from datetime import datetime,timezone
import importlib
import runpy
import sys
from types import FunctionType

from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader

ROOT=cold.ROOT
ALLOWED={
    'scripts.flow_matching.run_spectral_table2_gpu_memory_queue_v2',
    'scripts.flow_matching.run_registered_full_gpu_command_v2',
    'scripts.flow_matching.run_spectral_table3_gpu_memory_queue_v3',
    'scripts.flow_matching.run_spectral_table45_queue_v1',
    'scripts.flow_matching.run_crnngan_native_original_queue_v2',
    'scripts.flow_matching.run_flow_mismatching_full_image_evaluation_queue_v1',
}


def verify(config,name):
    if config['original_scope'] != dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273):
        raise ValueError('Preserve complete original research scope')
    for pin in config['source_receipts']:
        if cold.streaming_sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full-scope idle monitor handoff changed: '+pin['path'])
    role=config['roles'][name]
    if role['entry_module'] not in ALLOWED:
        raise ValueError('Only explicit registered original host entry modules supported')
    command=role['previous_owner']['command']
    index=command.index('-m')
    if command[index+1]!=role['entry_module'] or command[index+2:]!=role['original_arguments']:
        raise ValueError('Every original command argument must remain exact')
    queue=cold.original.read(ROOT/role['original_queue'])
    if len(queue.get(role['queue_items_key'],[]))!=role['required_original_slots']:
        raise ValueError('Every original complete queue slot must survive')
    return role,queue


def idle_allowed(role,state,output_exists,descendants):
    return (role=='original' and state.get('active_pid') is None
            and state.get('state') in {'waiting_for_memory','waiting_for_heavy_lock','waiting_full_gpu_resources'}
            and not state.get('shared_lock_owned')
            and all(cold.handoff.allowed_console_child(p['name'],p['exe']) for p in descendants))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--role',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=cold.original.read(ROOT/args.config)
    role,queue=verify(config,args.role)
    if args.probe_only:
        install_cached_loader(lambda snapshot:None)
        importlib.import_module(role['entry_module'])
        from scripts.flow_matching import run_spectral_original_queue as spectral
        import psutil
        api=spectral.guarded_api()
        memory=api['resource_snapshot']()
        print(dict(role=args.role,original_slots=role['required_original_slots'],original_arguments=role['original_arguments'],
                   every_source_and_queue_byte_pinned=True,cached_resource_snapshot=memory,
                   numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
                   private_bytes=psutil.Process().memory_info().private,original_numerical_child_executed=False))
        return
    base=ROOT/config['handoff_state_root']/args.role;base.mkdir(parents=True,exist_ok=True)
    owner=cold.cached_api()['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('The exact idle handoff already owned')
    try:
        function=FunctionType(cold.handoff.safe_handoff.__code__,dict(cold.handoff.__dict__,handoff_phase_allowed=idle_allowed))
        route=dict(role,runtime_configuration=args.config)
        function('original',route,base,lambda state:cold.original.write(base/'status.json',state),10)
        verify(config,args.role)
        recorded=[False]
        def record(memory):
            if not recorded[0]:
                cold.original.write(base/'actual_first_cached_telemetry.json',dict(snapshot=memory,original_guard_settings_unchanged=True,
                    captured_utc=datetime.now(timezone.utc).isoformat()))
                recorded[0]=True
        install_cached_loader(record)
        sys.argv=[str(ROOT/role['entry_module'].replace('.','/'))+'.py',*role['original_arguments']]
        runpy.run_module(role['entry_module'],run_name='__main__',alter_sys=True)
    finally:owner.close()


if __name__=='__main__':main()
