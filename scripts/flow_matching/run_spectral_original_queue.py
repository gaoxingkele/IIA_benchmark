"""Low-memory serial GPU queue for full author generation and evaluation."""
import argparse
import ctypes
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write
from scripts.flow_matching.run_light_controller import load_functions
from scripts.flow_matching.run_spectral_original_job import verify


def guarded_api():
    api=dict(globals())
    load_functions(ROOT,'scripts/flow_matching/run_tsad_queue.py',['exclusive_lock'],api)
    load_functions(ROOT,'scripts/flow_matching/heavy_resources.py',['resource_snapshot','has_headroom','terminate_tree'],api)
    load_functions(ROOT,'scripts/flow_matching/heavy_scheduler_v2.py',['resource_slot'],api)
    load_functions(ROOT,'scripts/flow_matching/run_grasp_protocol_queue.py',['gpu_snapshot','guarded_gpu_wait'],api)
    return api


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--preflight',required=True)
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    proof=read(ROOT/args.preflight)
    if (not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256']!=sha(ROOT/args.queue)
            or proof['full_author_model_configs_verified']!=14 or not proof['original_sampling_method_retained']):
        raise ValueError('Full original-interface preflight is missing')
    api=guarded_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('Original generation controller already live')
    state={'pid':os.getpid(),'queue_sha256':sha(ROOT/args.queue),'preflight_sha256':sha(ROOT/args.preflight),
        'jobs':{},'active_pid':None,'active_job':None}
    def update(values):
        state.update(values);write(base/'status.json',state)
    def verified_result(job):
        result=ROOT/job['output_directory']/'result.json'
        if not result.exists():return False
        command=[str(ROOT/queue['python']),'-X','utf8','-m','scripts.flow_matching.run_spectral_original_job',
                 '--queue',args.queue,'--job-id',job['id'],'--verify-result']
        process=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                               encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode!=0:write(base/'validation'/f"{job['id']}.json",{'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr})
        return process.returncode==0
    @contextmanager
    def ready_slot():
        while True:
            while api['gpu_snapshot'](queue['settings'])['gpu_free_bytes']<queue['settings']['minimum_gpu_free_bytes']:
                update({'state':'waiting_for_gpu_memory'});time.sleep(10)
            with api['resource_slot'](ROOT/queue['resource_lock'],queue['settings'],update):
                if api['gpu_snapshot'](queue['settings'])['gpu_free_bytes']>=queue['settings']['minimum_gpu_free_bytes']:
                    yield
                    return
            update({'state':'waiting_for_gpu_memory'});time.sleep(10)
    for job in queue['jobs']:
        if verified_result(job):
            state['jobs'][job['id']]='completed';continue
        if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
            state['jobs'][job['id']]='failed_or_partial_preserved';continue
        with ready_slot():
            stdout=base/'logs'/f"{job['id']}.stdout.log";stderr=base/'logs'/f"{job['id']}.stderr.log"
            stdout.parent.mkdir(exist_ok=True)
            with stdout.open('xb') as out,stderr.open('xb') as err:
                environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
                command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_spectral_original_job',
                         '--queue',args.queue,'--job-id',job['id']]
                child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update({'state':'running','active_job':job['id'],'active_pid':child.pid})
                code,peaks=api['guarded_gpu_wait'](child,queue['settings'],api,update)
        valid=verified_result(job) if code==0 else False
        state['jobs'][job['id']]='completed' if valid else 'failed_or_partial_preserved'
        write(base/'receipts'/f"{job['id']}.json",{'id':job['id'],'exitcode':code,'full_result_validated':valid,**peaks})
        update({'active_pid':None,'active_job':None,'state':'between_jobs'});time.sleep(2)
    update({'state':'completed' if all(v=='completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'})


if __name__=='__main__':main()
