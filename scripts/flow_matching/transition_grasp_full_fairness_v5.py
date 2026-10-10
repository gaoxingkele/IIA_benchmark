"""Replace only a verified childless scheduler after a completed full model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.grasp_protocol_runner import ROOT,complete,verify,sha
from scripts.flow_matching.transition_light_controllers_v2 import console_helper
from scripts.flow_matching.run_tsad_queue import exclusive_lock
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def boundary_ready(parent,state,expected,last_job,complete_check):
    if (parent.pid!=expected['pid'] or parent.create_time()!=expected['create_time']
            or parent.cmdline()!=expected['command'] or state.get('pid')!=parent.pid
            or state.get('queue_sha256')!=expected['queue_sha256']):return False
    if not last_job or last_job not in expected['job_ids']:return False
    if state.get('active_job') not in (None,last_job):return False
    if state.get('active_pid') and psutil.pid_exists(state['active_pid']):return False
    if any(not console_helper(c) for c in parent.children(recursive=True)):return False
    return bool(complete_check(last_job))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();path=ROOT/args.config;config=json.loads(path.read_text(encoding='utf-8'))
    for entry in config['source_receipts']:
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Full native handoff source differs')
    queue_path=ROOT/config['queue'];queue=json.loads(queue_path.read_text(encoding='utf-8'))
    if sha(queue_path)!=config['queue_sha256']:raise ValueError('Full native queue differs')
    verify(queue);jobs={j['id']:j for j in queue['jobs']}
    expected=dict(config['expected_controller'],queue_sha256=sha(queue_path),job_ids=set(jobs))
    parent=psutil.Process(expected['pid'])
    if parent.create_time()!=expected['create_time'] or parent.cmdline()!=expected['command']:
        raise ValueError('Corresponding guarded controller identity differs')
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'transition.json').exists():raise FileExistsError('Previous native handoff remains intact')
    owner=exclusive_lock(base/'controller.lock')
    if owner is None:raise RuntimeError('Corresponding native handoff already live')
    state_path=ROOT/queue['state_root']/'status.json';last_job=None
    def full(job):return complete(jobs[job])
    try:
        while True:
            if parent.create_time()!=expected['create_time'] or parent.cmdline()!=expected['command']:
                raise ValueError('Original scheduler identity changed during observation')
            state=json.loads(state_path.read_text(encoding='utf-8'))
            if state.get('active_job'):last_job=state['active_job']
            observed=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='waiting_full_model_boundary',
                original_pid=parent.pid,original_active_pid=state.get('active_pid'),original_active_job=state.get('active_job'),
                last_observed_job=last_job,config_sha256=sha(path))
            write(base/'status.json',observed)
            child_pid=state.get('active_pid')
            if child_pid and psutil.pid_exists(child_pid):
                child=psutil.Process(child_pid);command=child.cmdline()
                if (child.ppid()!=parent.pid or 'scripts.flow_matching.grasp_protocol_cuda_runtime' not in command
                        or last_job not in command):raise ValueError('Observed full native child differs')
                try:child.wait(timeout=30)
                except psutil.TimeoutExpired:continue
            if not boundary_ready(parent,state,expected,last_job,full):time.sleep(.1);continue
            parent.suspend();stopped=False
            try:
                again=json.loads(state_path.read_text(encoding='utf-8'))
                if not boundary_ready(parent,again,expected,last_job,full):continue
                saved=dict(original_pid=parent.pid,original_create_time=parent.create_time(),original_command=parent.cmdline(),
                    last_full_job=last_job,last_state=again,original_guard_active_until_full_model_exit=True,
                    model_processes_stopped=False,full_queue_sha256=sha(queue_path),full_jobs_and_budgets_unchanged=True,
                    config_sha256=sha(path))
                write(base/'before_handoff.json',saved)
                parent.terminate();parent.wait(timeout=10);stopped=True
            finally:
                if not stopped and parent.is_running():parent.resume()
            if stopped:break
        with (base/'successor.stdout.log').open('xb') as out,(base/'successor.stderr.log').open('xb') as err:
            child=subprocess.Popen(config['successor_command'],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        handle=psutil.Process(child.pid)
        write(base/'transition.json',dict(saved,new_pid=child.pid,new_create_time=handle.create_time(),
            new_command=config['successor_command'],handoff_complete=True,captured_utc=datetime.now(timezone.utc).isoformat()))
        write(base/'status.json',dict(observed,state='completed_full_boundary_handoff',new_pid=child.pid))
        print(json.dumps(dict(handoff_complete=True,new_pid=child.pid,last_full_job=last_job)))
    finally:owner.close()


if __name__=='__main__':main()
