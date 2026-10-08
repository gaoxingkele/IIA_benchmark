"""Execute every validated case and keep unvalidated complete-budget cases pending."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.heavy_resources import waited_lock,wait_for_headroom,guarded_wait
from scripts.flow_matching.prepare_tsad_execution import sha,write_json
from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts


def native_proof(root,job,settings):
    parent=root/settings['preflight_parent_queue']
    if sha(parent)!=settings['preflight_parent_sha256']:raise ValueError('Native preflight queue changed')
    original=json.loads(parent.read_text());binding=next(j for j in original['jobs'] if j['id']==job['id'])
    for key in ('model_parameters','train_parameters','dataset','activation_checkpointing','release_checkpoint','raw_sha256'):
        assert job.get(key)==binding.get(key),key
    name=job['dataset']+('_row'+str(job['ablation_row']) if 'ablation_row' in job else '')
    path=root/'docs/reports'/('fm_crossad_complete_preflight_v4_'+name+'_2026-10-09.json')
    if not path.exists():return None
    report=json.loads(path.read_text());assert report['queue_sha256']==sha(parent) and len(report['records'])==1
    record=report['records'][0]
    assert record['dataset']==job['dataset'] and record['finite_backward'] and record['deterministic_release_inference']
    assert record['native_batch_shape'][:2]==[job['train_parameters']['batch_size'],job['model_parameters']['seq_len']]
    return {'path':path.relative_to(root).as_posix(),'sha256':sha(path),'native_batch_shape':record['native_batch_shape']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True);args=parser.parse_args()
    path=ROOT/args.queue;queue=json.loads(path.read_text());settings=queue['settings'];base=ROOT/settings['output_root'];base.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'queue_sha256':sha(path),'active_job':None,'active_pid':None,'jobs':{}}
    def update(value):state.update(value);write_json(base/'status.json',state)
    with waited_lock(base/'worker.lock',update):
        frozen={r['path']:r['sha256'] for j in queue['jobs'] for r in j['frozen_files']}
        for name,expected in frozen.items():assert sha(ROOT/name)==expected,name
        update({'state':'starting','frozen_files_verified':len(frozen)})
        while True:
            for job in queue['jobs']:
                output=ROOT/job['output_directory'];final=output/'result.json'
                if final.exists():
                    receipt=json.loads(final.read_text());assert receipt['experiment_sha256']==hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
                    verify_artifacts(receipt);state['jobs'][job['id']]='completed';continue
                if (output/'workspace').exists() or (output/'failure.json').exists():state['jobs'][job['id']]='failed_or_partial_preserved';continue
                proof=native_proof(ROOT,job,settings)
                if proof is None:state['jobs'][job['id']]='awaiting_full_native_batch_preflight';continue
                with waited_lock(ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock',update):
                    start=wait_for_headroom(settings,update);output.mkdir(parents=True,exist_ok=True)
                    write_json(output/'native_preflight_binding.json',proof)
                    env={k.upper():v for k,v in os.environ.items()};env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(settings['cpu_threads']),MKL_NUM_THREADS=str(settings['cpu_threads']))
                    command=[str(ROOT/settings['python']),'-u',str(ROOT/job['runner']),'--queue',str(path),'--job-id',job['id']]
                    flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                    with (output/'stdout.log').open('a',encoding='utf-8') as out,(output/'stderr.log').open('a',encoding='utf-8') as err:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
                        update({'state':'running','active_job':job['id'],'active_pid':child.pid,'active_stage':'full_author_pipeline'})
                        code,usage=guarded_wait(child,settings,update)
                    write_json(output/'resource_usage.json',dict(usage,start_resources=start))
                    if code or not final.exists():
                        write_json(output/'failure.json',{'status':'failed_preserved','exit_code':code,'resources':usage});state['jobs'][job['id']]='failed_or_partial_preserved'
                    else:verify_artifacts(json.loads(final.read_text()));state['jobs'][job['id']]='completed'
                    update({'active_job':None,'active_pid':None,'active_stage':None})
                # Give the waiting recovery/preflight producers a chance to acquire the shared lock.
                time.sleep(2)
            pending=[j for j in queue['jobs'] if state['jobs'].get(j['id'])=='awaiting_full_native_batch_preflight']
            if not pending:
                update({'state':'completed' if all(state['jobs'].get(j['id'])=='completed' for j in queue['jobs']) else 'completed_with_unresolved_jobs'});break
            update({'state':'waiting_for_remaining_native_preflight','pending_native_cases':len(pending)});time.sleep(30)


if __name__=='__main__':main()
