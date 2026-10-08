"""Guarded exact-author CrossAD incident recovery, using original child runner."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import psutil
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def verify(config):
    for source in config['source_receipts']:
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Frozen recovery source changed: '+source['path'])
    native=json.loads((ROOT/config['native_queue']).read_text(encoding='utf-8'))
    if native['jobs']!=[c['job'] for c in config['jobs']]:raise ValueError('Native dispatch differs from recovery roster')
    for case in config['jobs']:
        parent=json.loads((ROOT/case['original_queue']).read_text(encoding='utf-8'))
        original=next(j for j in parent['jobs'] if j['id']==case['original_id'])
        if original!=case['original_job']:raise ValueError('Original job binding changed')
        if {k:v for k,v in case['job'].items() if k not in ('id','output_directory')}!={k:v for k,v in original.items() if k not in ('id','output_directory')}:
            raise ValueError('Recovery changed experiment parameters')
        if native['settings']!=parent['settings'] or case['settings']!=parent['settings']:
            raise ValueError('Recovery changed author protocol')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();path=ROOT/args.queue;config=json.loads(path.read_text(encoding='utf-8'))
    verify(config)
    if args.verify_only:print(json.dumps({'verified_jobs':len(config['jobs'])}));return
    runtime=json.loads((ROOT/config['runtime_config']).read_text(encoding='utf-8'))
    controller,namespace,_=load_controller(ROOT,runtime,'crossad')
    base=ROOT/config['output_root'];base.mkdir(parents=True,exist_ok=True)
    lock=namespace['exclusive_lock'](base/'worker.lock')
    if lock is None:raise RuntimeError('Recovery controller already live')
    state={'pid':os.getpid(),'queue_sha256':digest(path),'active_pid':None,'active_job':None,'jobs':{}}
    def update(values):state.update(values);namespace['write_json'](base/'status.json',state)
    for case in config['jobs']:
        job=case['job'];output=ROOT/job['output_directory'];final=output/'result.json'
        if final.exists():
            receipt=json.loads(final.read_text(encoding='utf-8'))
            assert receipt['experiment_sha256']==hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
            namespace['verify_artifacts'](receipt);state['jobs'][job['id']]='completed';continue
        if output.exists():state['jobs'][job['id']]='failed_or_partial_preserved';continue
        proof=controller.native_proof(ROOT,case['original_job'],case['settings'])
        if proof is None:raise RuntimeError('Original full-native proof missing; recovery not launched')
        live=[p.pid for p in psutil.process_iter(['pid','cmdline']) if case['original_id'] in (p.info['cmdline'] or [])]
        if live:raise RuntimeError('Matching original model still active: '+str(live))
        with namespace['resource_slot'](ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock',config['settings'],update) as resources:
            output.mkdir(parents=True,exist_ok=True)
            namespace['write_json'](output/'native_preflight_binding.json',proof)
            command=[str(ROOT/case['settings']['python']),'-u',str(ROOT/job['runner']),
                     '--queue',str(ROOT/config['native_queue']),'--job-id',job['id']]
            env={k.upper():v for k,v in os.environ.items()}
            env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',CUDA_VISIBLE_DEVICES='-1',
                       OMP_NUM_THREADS=str(case['settings']['cpu_threads']),MKL_NUM_THREADS=str(case['settings']['cpu_threads']))
            flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
            with (output/'stdout.log').open('a',encoding='utf-8') as out,(output/'stderr.log').open('a',encoding='utf-8') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
                update({'state':'running','active_job':job['id'],'active_pid':child.pid})
                code,usage=namespace['guarded_wait'](child,config['settings'],update)
            namespace['write_json'](output/'resource_usage.json',dict(usage,start_resources=resources))
            if code or not final.exists():
                namespace['write_json'](output/'failure.json',{'exit_code':code,'resources':usage,'original_failure_preserved':case['original_failure_path']})
                state['jobs'][job['id']]='failed_or_partial_preserved'
            else:
                receipt=json.loads(final.read_text(encoding='utf-8'))
                assert receipt['experiment_sha256']==hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
                namespace['verify_artifacts'](receipt);state['jobs'][job['id']]='completed'
            update({'active_job':None,'active_pid':None})
        time.sleep(2)
    update({'state':'completed' if all(v=='completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'})


if __name__=='__main__':main()
