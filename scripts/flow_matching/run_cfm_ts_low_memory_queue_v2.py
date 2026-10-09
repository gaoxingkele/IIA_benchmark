"""Run the frozen full CFM-TS jobs with isolated result verification.

NumPy/Torch stay in short-lived verifier or model children. Both original and
exact-budget recovery queues share a family slot with the original safeguards.
"""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.run_light_controller import ROOT,load_functions

load_functions(ROOT,'scripts/flow_matching/prepare_cfm_ts_original_data.py',['sha','read'],globals())
load_functions(ROOT,'scripts/flow_matching/run_cfm_ts_original_job.py',['verify','write'],globals())


def resource_api():
    namespace=dict(globals())
    load_functions(ROOT,'scripts/flow_matching/run_tsad_queue.py',['exclusive_lock'],namespace)
    load_functions(ROOT,'scripts/flow_matching/heavy_resources.py',
        ['resource_snapshot','has_headroom','terminate_tree','guarded_wait'],namespace)
    load_functions(ROOT,'scripts/flow_matching/heavy_scheduler_v2.py',['resource_slot'],namespace)
    return namespace


def binding(runtime,name):
    for receipt in runtime['source_receipts']:
        if sha(ROOT/receipt['path'])!=receipt['sha256']:
            raise ValueError('Frozen controller source differs: '+receipt['path'])
    selected=runtime['controllers'][name]
    if sha(ROOT/selected['queue'])!=selected['queue_sha256']:
        raise ValueError('Frozen full-scope queue differs')
    queue=read(ROOT/selected['queue']);verify(queue)
    return selected,queue


def external_complete(queue_path,queue,job):
    if not (ROOT/job['output_directory']/'result.json').exists():return False
    script=("import json,sys;from scripts.flow_matching.run_cfm_ts_original_job import ROOT,read,complete,verify;"
        "q=read(ROOT/sys.argv[1]);verify(q);j=next(j for j in q['jobs'] if j['id']==sys.argv[2]);"
        "v=complete(j);print(json.dumps({'complete':v}));sys.exit(0 if v else 3)")
    result=subprocess.run([str(ROOT/queue['python']),'-X','utf8','-c',script,queue_path,job['id']],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode not in (0,3):
        raise ValueError('Frozen full artifact verification failed: '+result.stderr[-2000:])
    return result.returncode==0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--controller',choices=['original','recovery'],default='original')
    parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--probe-completed-id')
    args=parser.parse_args();runtime=read(ROOT/args.runtime_config);selected,queue=binding(runtime,args.controller)
    if args.probe_only:
        verified=None
        if args.probe_completed_id:
            job=next(j for j in queue['jobs'] if j['id']==args.probe_completed_id)
            verified=external_complete(selected['queue'],queue,job)
        print(json.dumps({'private_bytes':psutil.Process().memory_info().private,
            'numpy_imported':'numpy' in sys.modules,'torch_imported':'torch' in sys.modules,
            'registered_jobs':len(queue['jobs']),'selected_jobs':len(selected.get('selected_job_ids',queue['jobs'])),
            'full_completed_artifact_verified':verified}));return
    api=resource_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('CFM-TS lane already has a controller')
    state={'pid':os.getpid(),'queue_sha256':sha(ROOT/selected['queue']),'runtime_sha256':sha(ROOT/args.runtime_config),
        'jobs':{},'active_pid':None,'active_job':None}
    def update(values):state.update(values);write(base/'status.json',state)
    jobs=[j for j in queue['jobs'] if 'selected_job_ids' not in selected or j['id'] in selected['selected_job_ids']]
    try:
        for job in jobs:
            if external_complete(selected['queue'],queue,job):
                state['jobs'][job['id']]='completed';continue
            if (ROOT/job['output_directory']).exists() or (base/'receipts'/f"{job['id']}.json").exists():
                state['jobs'][job['id']]='failed_or_partial_preserved';continue
            with api['resource_slot'](ROOT/runtime['family_lock'],queue['resource_policy'],update):
                logs=base/'logs';logs.mkdir(exist_ok=True)
                command=[str(ROOT/queue['python']),'-u','-m','scripts.flow_matching.run_cfm_ts_original_job',
                    '--queue',selected['queue'],'--job-id',job['id']]
                with (logs/(job['id']+'.stdout.log')).open('xb') as out,(logs/(job['id']+'.stderr.log')).open('xb') as err:
                    worker=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    update({'active_pid':worker.pid,'active_job':job['id'],'state':'running'})
                    code,peaks=api['guarded_wait'](worker,queue['resource_policy'],update=update)
                done=external_complete(selected['queue'],queue,job)
            state['jobs'][job['id']]='completed' if done else 'failed_or_partial_preserved'
            write(base/'receipts'/f"{job['id']}.json",{'exit_code':code,'complete':done,'resource_observations':peaks})
            update({'active_pid':None,'active_job':None,'state':'between_jobs'})
        update({'state':'completed' if all(v=='completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'})
    finally:lane.close()


if __name__=='__main__':main()
