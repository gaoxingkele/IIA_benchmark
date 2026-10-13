"""Safely replace only an idle metadata retry parent with queued family admission.

Original full285 slots, both full400 retries, model entry, numeric code and all
resource limits remain identical. The active original CFM worker is never stopped.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import FunctionType
from scripts.flow_matching import run_cfm_metadata_permission_full_retry_v1 as previous
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import queued_family_lock_v1 as queued

ROOT=cold.ROOT


def verify(config):
    for pin in config['source_receipts']:
        if cold.streaming_sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen queued admission source changed')
    runtime=cold.original.read(ROOT/config['original_runtime']);chosen,queue=previous.verify_runtime(runtime)
    if len(chosen)!=2 or len(queue['jobs'])!=285 or runtime['family_lock']!=config['family_lock']:
        raise ValueError('Same family lock and both full retries in unchanged285 scope required')
    return runtime,queue


def namespace(runtime,queue):
    values=previous.namespace();old_api=values['resource_api']
    def api():
        result=old_api();oldslot=result['resource_slot'];family=(ROOT/runtime['family_lock']).resolve()
        @contextmanager
        def selected(path,settings,update=None,poll_seconds=10):
            if Path(path).resolve()!=family:
                with oldslot(path,settings,update,poll_seconds) as snapshot:yield snapshot
                return
            if settings!=queue['resource_policy']:raise ValueError('Original admission/emergency policy required')
            with queued.resource_slot(path,settings,result,update,poll_seconds) as snapshot:yield snapshot
        result['resource_slot']=selected;return result
    values['resource_api']=api
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=cold.original.read(ROOT/args.config);runtime,queue=verify(config)
    if args.probe_only:
        print(dict(original_slots=285,full_retry_jobs=2,epochs=[400,400],updates=[6400,6400],
            original_guards_and_numeric_child_unchanged=True,training_executed=False));return
    base=ROOT/config['handoff_state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=cold.cached_api()['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Queued admission handoff already owned')
    try:
        expected=config['previous_owner'];statepath=(ROOT/config['previous_state_path']).resolve()
        def read(path):
            state=cold.handoff.read(path)
            if Path(path).resolve()==statepath:
                selected=runtime['controllers']['recovery']
                if (state.get('pid')!=expected['pid'] or state.get('queue_sha256')!=selected['queue_sha256']
                        or state.get('runtime_sha256')!=cold.streaming_sha(ROOT/config['original_runtime'])):
                    raise ValueError('Legacy exact owner/queue/runtime binding changed')
                if state.get('create_time') is None:state=dict(state,create_time=expected['create_time'])
            return state
        def idle(role,state,output_exists,descendants):
            return role=='original' and state.get('active_pid') is None and not state.get('shared_lock_owned') and state.get('state') in {'waiting_for_heavy_lock','waiting_for_memory','between_jobs'} and all(cold.handoff.allowed_console_child(p['name'],p['exe']) for p in descendants)
        fn=FunctionType(cold.handoff.safe_handoff.__code__,dict(cold.handoff.__dict__,read=read,handoff_phase_allowed=idle))
        fn('original',dict(previous_owner=expected,previous_state_path=config['previous_state_path'],runtime_configuration=args.config),base,lambda s:cold.original.write(base/'status.json',s),1)
        verify(config)
        values=namespace(runtime,queue);saved=sys.argv
        oldwrite=values['write']
        def write(path,data):
            import os,psutil
            if Path(path).resolve()==statepath and isinstance(data,dict) and data.get('pid')==os.getpid():
                data=dict(data,create_time=psutil.Process().create_time(),queued_admission_configuration=args.config)
            return oldwrite(path,data)
        values['write']=write
        try:
            sys.argv=[saved[0],'--runtime-config',config['original_runtime'],'--controller','recovery'];values['main']()
        finally:sys.argv=saved
    finally:owner.close()


if __name__=='__main__':main()
