"""Independently audit registered complete GRASP arrays under the shared CPU slot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();path=ROOT/args.config;config=read(path)
    for source in config['source_receipts']:
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Frozen full audit source differs')
    queue_path=ROOT/config['queue'];queue=read(queue_path)
    if digest(queue_path)!=config['queue_sha256']:raise ValueError('Canonical native queue differs')
    records=[]
    for record in config['full_results']:
        job=next(j for j in queue['jobs'] if j['id']==record['id'])
        result_path=ROOT/job['output_directory']/'result.json'
        if digest(result_path)!=record['result_sha256']:raise ValueError('Registered full result changed')
        records.append(dict(record,seed=job['seed'],status='completed',result_path=result_path.relative_to(ROOT).as_posix()))
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Audited full result captures remain immutable')
    if args.worker:
        from scripts.flow_matching.grasp_protocol_runner import complete,verify
        from scripts.flow_matching.audit_original_native_execution import audit_grasp_arrays
        verify(queue)
        for record in records:
            job=next(j for j in queue['jobs'] if j['id']==record['id'])
            if not complete(job):raise ValueError('All full epochs, updates and artifacts are required')
        audited=audit_grasp_arrays(ROOT,queue,dict(jobs=records))
        if len(audited)!=len(records):raise ValueError('Not all registered full results independently audited')
        target.mkdir(parents=True)
        report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),queue=config['queue'],queue_sha256=digest(queue_path),
            audited_full_results=audited,full_result_audit_passed=True,
            all_original_experiments_complete=False,additional_training_seed_slots=0,
            boundary='Complete original native training artifacts; independent full validation/test row and label identity, saved checkpoint binding and every score-recipe metric recomputed. One original seed per ablation, no seed uncertainty or paper-equivalence claim.')
        write(target/'independent_array_audit.json',report)
        write(target/'validation.json',dict(source_receipts=config['source_receipts'],
            outputs={'independent_array_audit.json':digest(target/'independent_array_audit.json')}))
        print(json.dumps(dict(audited_full_results=len(audited),epochs=[a['epochs_completed'] for a in audited])));return
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'stdout.log').exists():raise FileExistsError('Previous full audit attempt must remain intact')
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Corresponding full audit already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),config_sha256=digest(path),state='starting',active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.audit_registered_grasp_results_v1',
                '--config',args.config,'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid);update(dict(state='auditing',active_pid=child.pid,
                    active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage=api['guarded_wait'](child,config['settings'],update)
        proof=target/'independent_array_audit.json'
        valid=bool(code==0 and proof.exists() and read(proof)['full_result_audit_passed'])
        write(base/'receipt.json',dict(exit_code=code,full_audit_passed=valid,**usage))
        update(dict(state='completed' if valid else 'failed_preserved',active_pid=None,full_audit_passed=valid))
        if not valid:raise RuntimeError('Complete native independent audit failed; preserve evidence')
    finally:owner.close()


if __name__=='__main__':main()
