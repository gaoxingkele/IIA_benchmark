"""Retire four exact idle metadata owners, preserving every numerical worker."""
from datetime import datetime,timezone
import os
import subprocess
import time
import psutil

from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import ROOT,read,sha,write,verify
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot


def main():
    configpath='configs/runtime/fm_anomaly_transformer_cold_resource_entry.v2.json'
    config=read(ROOT/configpath);records=[];before_system=resource_snapshot()
    for role in ['author','restored','paper','sensitivity']:
        selected=verify(config,role);expected=selected['previous_owner']
        original=read(ROOT/selected['queue']);statepath=ROOT/original['state_root']/'status.json'
        process=psutil.Process(expected['pid'])
        def inspect_idle():
            if process.create_time()!=expected['create_time'] or process.cmdline()!=expected['command']:
                raise RuntimeError('Exact previous metadata identity changed')
            state=read(statepath)
            if (state.get('pid')!=process.pid or state.get('active_pid') is not None or state.get('shared_lock_owned')
                or state.get('state') not in ['waiting_for_memory','waiting_for_heavy_lock']):
                raise RuntimeError('Controller is not proven idle')
            children=[]
            for child in process.children(recursive=True):
                entry=dict(pid=child.pid,name=child.name(),exe=child.exe(),create_time=child.create_time())
                if child.name().lower()!='conhost.exe' or not child.exe().lower().endswith('\\windows\\system32\\conhost.exe'):
                    raise RuntimeError('A numerical or unknown child prevents the idle handoff')
                children.append(entry)
            return dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline(),
                        private_bytes=process.memory_info().private,state_at_check=state,only_console_children=children)
        before=inspect_idle();process.suspend();stopped=False
        try:
            suspended=inspect_idle()
            process.terminate();process.wait(timeout=15);stopped=True
        finally:
            if not stopped and process.is_running():process.resume()
        output=ROOT/config['state_root']/role;output.mkdir(parents=True,exist_ok=True)
        command=[config['python'],'-B','-X','utf8','-u','-m',
            'scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1','--config',configpath,'--role',role]
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                         PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'))
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        successor=psutil.Process(child.pid);ct=successor.create_time();deadline=time.monotonic()+40
        while True:
            if child.poll() is not None:raise RuntimeError('Successor exited; inspect preserved logs, do not blindly restart')
            current=read(statepath)
            observed=output/'actual_first_cached_snapshot.json'
            if current.get('pid')==child.pid and observed.exists():break
            if time.monotonic()>deadline:raise TimeoutError('Observation expired; successor may still run, do not restart')
            time.sleep(.25)
        first=read(observed)
        if any(first[k] for k in ['numpy_imported','pandas_imported','torch_imported']):
            raise RuntimeError('Successor loaded numerical libraries; preserve it for inspection')
        record=dict(role=role,previous_exact_idle_owner=before,suspended_recheck=suspended,
            successor_pid=child.pid,successor_create_time=ct,successor_command=successor.cmdline(),
            successor_state=current,successor_first_snapshot=first,
            released_private_commit_bytes=before['private_bytes']-first['private_bytes'],
            numerical_children_stopped=0,numerical_settings_and_guards_unchanged=True)
        records.append(record);write(output/'handoff_receipt.json',record)
    write(ROOT/config['state_root']/'handoff_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
     configuration=configpath,configuration_sha256=sha(ROOT/configpath),records=records,
     observed_system_before=before_system,observed_system_after=resource_snapshot(),
     sum_released_controller_private_commit_bytes=sum(r['released_private_commit_bytes'] for r in records),
     numerical_children_stopped=0,all_original_experiments_complete=False))
    print(__import__('json').dumps(dict(replaced_idle_metadata_owners=4,
     released_private_commit_GiB=sum(r['released_private_commit_bytes'] for r in records)/1024**3,
     successors=[dict(role=r['role'],pid=r['successor_pid'],create_time=r['successor_create_time'],
                     private_MiB=r['successor_first_snapshot']['private_bytes']/1024**2) for r in records],
     numerical_children_stopped=0)))


if __name__=="__main__":main()
