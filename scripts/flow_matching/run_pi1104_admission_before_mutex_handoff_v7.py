"""Release an idle memory-waiting owner; admit before the same bounded CPU lock.

Keep all13 incident records, the selected full Pi source1104 retry, original40
GiB admission and original numerical child/caps. Reuse the existing proven
resource_slot recheck and reserved-headroom binding. No native worker is stopped.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import FunctionType

from scripts.flow_matching import run_selected_incident_source_recovery_v6 as previous
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader

ROOT=previous.ROOT


def reserve_before_lock(namespace,queue,lock,api):
    old=namespace['waited_lock']
    reservation={'snapshot':None}
    @contextmanager
    def selected(path,waiting=None,poll_seconds=10):
        if Path(path).resolve()!=Path(lock).resolve():
            with old(path,waiting,poll_seconds) as handle:yield handle
            return
        with api['resource_slot'](path,queue['settings'],waiting,poll_seconds) as snapshot:
            reservation['snapshot']=snapshot
            try:yield snapshot
            finally:reservation['snapshot']=None
    def headroom(settings,update=None,poll_seconds=10):
        if settings!=queue['settings'] or reservation['snapshot'] is None:
            raise RuntimeError('Headroom requires the original selected settings and owned resource slot')
        return dict(reservation['snapshot'])
    namespace.update(waited_lock=selected,wait_for_headroom=headroom)
    return reservation


def verify(config):
    for pin in config['source_receipts']:
        if cold.streaming_sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen idle admission handoff changed: '+pin['path'])
    registration=cold.original.read(ROOT/config['original_runtime'])
    function,queue,binding=previous.bind_source(registration)
    if (len(queue['jobs'])!=13 or queue['settings']['minimum_commit_headroom_bytes']!=40*2**30
            or registration['resource_lock']!=config['resource_lock']
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('Full13 incident scope, unchanged40GiB admission and same original mutex required')
    case=next(c for c in queue['jobs'] if c['job']['id']==registration['selected_job_id'])
    original=cold.original.read(ROOT/case['original_queue'])
    if len(original['jobs'])!=1140 or original['settings']['seeds']!=[42,1103,1104,1105,1106,1107]:
        raise ValueError('All1140 original Pi jobs and six original training seeds must remain')
    if (case['original_id']!='pi_author__source_runtime__psm__full__seed1104'
            or case['job']['author_config']['training']['n_epochs']!=3
            or case['job']['author_config']['model']['d_model']!=512
            or case['job']['author_config']['data']['batch_size']!=256
            or (ROOT/case['job']['output_directory']).exists()):
        raise ValueError('Exact full original unstarted source1104 retry must remain')
    return registration,function,queue,binding,case


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=cold.original.read(ROOT/args.config)
    install_cached_loader(lambda snapshot:None)
    registration,function,queue,binding,case=verify(config)
    if args.probe_only:
        print(dict(original_incident_cases=13,original_Pi_jobs=1140,selected_retry_cases=1,
                   unchanged_admission_bytes=queue['settings']['minimum_commit_headroom_bytes'],
                   original_worker='scripts.flow_matching.run_resource_recovery_job',
                   numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules));return
    base=ROOT/config['handoff_state_root'];base.mkdir(parents=True,exist_ok=True)
    api=cold.cached_api();owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Admission-before-mutex handoff already owned')
    try:
        expected=config['previous_owner']
        statepath=(ROOT/config['previous_state_path']).resolve()
        original_read=cold.handoff.read
        def legacy_state_read(path):
            state=original_read(path)
            if Path(path).resolve()==statepath:
                if state.get('pid')!=expected['pid'] or state.get('queue_sha256')!=cold.streaming_sha(ROOT/registration['queue']):
                    raise ValueError('Legacy state PID/whole-queue fingerprint changed')
                if state.get('create_time') is None:
                    # In-memory schema adapter only. safe_handoff independently
                    # checks actual PID/create_time/full command twice around suspend.
                    state=dict(state,create_time=expected['create_time'])
            return state
        def idle(role,state,output_exists,descendants):
            return (role=='original' and state.get('active_pid') is None and state.get('state')=='waiting_for_memory'
                    and not state.get('jobs') and not output_exists
                    and all(cold.handoff.allowed_console_child(p['name'],p['exe']) for p in descendants))
        handoff=FunctionType(cold.handoff.safe_handoff.__code__,dict(cold.handoff.__dict__,read=legacy_state_read,handoff_phase_allowed=idle))
        route=dict(previous_owner=expected,previous_state_path=config['previous_state_path'],
                   output_directory=case['job']['output_directory'],runtime_configuration=args.config)
        handoff('original',route,base,lambda state:cold.original.write(base/'status.json',state),10)
        registration,function,queue,binding,case=verify(config)
        reservation=reserve_before_lock(function.__globals__,queue,ROOT/config['resource_lock'],api)
        sys.argv=[binding['source_path'],'--queue',registration['queue']]
        function()
    finally:owner.close()


if __name__=='__main__':main()
