"""Retire only the proven-idle strict CPU adapter and start its capped successor."""
import argparse
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess

import psutil

from scripts.flow_matching.run_bounded_strict_cpu_queue_v2 import ROOT, bind, digest
from scripts.flow_matching.run_tsad_queue import complete_result


def verify_idle(registration,state,observed,previous_terminal):
    old=registration['predecessor']
    if (observed['pid']!=old['pid'] or observed['create_time']!=old['create_time'] or observed['command']!=old['command']
            or state['pid']!=old['pid'] or state['create_time']!=old['create_time']
            or state['runtime_sha256']!=registration['parent_resource_registration_sha256']
            or state['queue_sha256']!=registration['controllers']['strict_cpu']['queue_sha256']
            or state['state']!='waiting_for_heavy_lock' or state.get('active_pid') is not None
            or state.get('shared_lock_owned') is not False or state['requested_job']!=registration['requested_job']
            or state['requested_command']!=registration['requested_command']
            or any(c['name'].lower()!='conhost.exe' for c in observed['children'])):
        raise ValueError('Exact owned idle controller with no native child required')
    if state.get('active_job') is not None:
        if (previous_terminal['job_id']!=state['active_job'] or previous_terminal['job_id']==state['requested_job']
                or previous_terminal['exit_code']!=0 or previous_terminal['resource_guard_aborted']
                or not previous_terminal['full_registered_budget_retained']
                or previous_terminal['queue_sha256']!=state['queue_sha256']):
            raise ValueError('Stale previous-job field requires successful original terminal proof')
    return dict(actual_process=observed,original_status=state,previous_terminal=previous_terminal,verified_idle=True)


def observation(process):
    return dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline(),
                children=[dict(pid=p.pid,create_time=p.create_time(),name=p.name()) for p in process.children(recursive=True)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    args=parser.parse_args()
    registration=json.loads((ROOT/args.runtime_config).read_text(encoding='utf-8'))
    bind(registration)
    state_path=ROOT/registration['predecessor_status']
    def verify():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        previous=json.loads((ROOT/registration['previous_terminal_receipt']).read_text(encoding='utf-8'))
        process=psutil.Process(registration['predecessor']['pid'])
        record=verify_idle(registration,state,observation(process),previous)
        queue=json.loads((ROOT/registration['controllers']['strict_cpu']['queue']).read_text(encoding='utf-8'))
        old_job=next(j for j in queue['jobs'] if j['id']==previous['job_id'])
        if not complete_result(ROOT,old_job):raise ValueError('Previous full model completion must independently verify')
        job=next(j for j in queue['jobs'] if j['id']==state['requested_job'])
        if (ROOT/job['output_directory']).exists():raise ValueError('Do not overwrite an existing pending attempt')
        return process,record
    process,record=verify()
    report=ROOT/registration['report_directory'];report.mkdir(parents=True,exist_ok=True)
    receipt_path=report/'handoff_receipt.json'
    if receipt_path.exists():raise FileExistsError('Preserve original handoff evidence')
    snapshot=report/'preserved_idle_cpu_status.json'
    if snapshot.exists():raise FileExistsError('Preserve original idle status')
    shutil.copyfile(state_path,snapshot)
    process,rechecked=verify()
    process.terminate()
    code=process.wait(timeout=10)
    command=[registration['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_bounded_strict_cpu_queue_v2','--runtime-config',args.runtime_config]
    with (report/'controller.stdout.log').open('xb') as out,(report/'controller.stderr.log').open('xb') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        actual=psutil.Process(child.pid)
        launched=dict(pid=actual.pid,create_time=actual.create_time(),command=actual.cmdline())
    receipt=dict(captured_utc=datetime.now(timezone.utc).isoformat(),runtime=args.runtime_config,runtime_sha256=digest(ROOT/args.runtime_config),
                 retired_verified_idle_controller=rechecked,old_terminal_exit_code=code,preserved_idle_status_sha256=digest(snapshot),
                 launched_controller=launched,registered_original_jobs=175,original_cpu_threads=2,
                 original_model_workers_stopped=0,additional_training_seed_slots=0,new_performance_rows=0,all_original_experiments_complete=False)
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(retired_verified_idle_controller=1,full_original_cpu_jobs=175,new_controller=launched,model_workers_stopped=0)))


if __name__=='__main__':main()
