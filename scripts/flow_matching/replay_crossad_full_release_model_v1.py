"""Strictly replay original complete released weights with every native evaluator."""
import argparse
import json
import os
import subprocess

import psutil

from scripts.flow_matching import replay_crossad_full_native_model_v1 as original
from scripts.flow_matching.replay_crossad_full_native_model_v2 import restore_original_profile_buffers

ROOT,read,digest=original.ROOT,original.read,original.digest


def worker(config,path):
    import iia_benchmark.models.fm_crossad_memory_v2 as native
    queue=read(config['queue']);job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    from scripts.flow_matching.audit_crossad_full_release_completion_v1 import release_only
    release_only(job,read(ROOT/job['output_directory']/'result.json'))
    factory=native.build_memory_model;restored=[]
    def faithful_factory(*args,**kwargs):
        model=factory(*args,**kwargs)
        def restore(module,state,prefix,metadata,strict,missing,unexpected,errors):
            if prefix:raise ValueError('Restore only original root state dictionary metadata')
            restored.extend(restore_original_profile_buffers(module,state))
        model.register_load_state_dict_pre_hook(restore)
        return model
    native.build_memory_model=faithful_factory
    original.worker(config,path)
    target=ROOT/config['artifact_directory'];proof=read(target/'full_native_model_replay.json')
    proof.pop('required_original_main_training_seeds')
    proof.update(original_THOP_profile_buffers_restored_without_dropping_algorithm_weights=restored,
        original_full_state_dictionary_loaded_strictly=True,fixed_release_checkpoint_evaluation=True,
        registered_required_release_evaluations=1,completed_training_seed_slots=0,
        original_release_checkpoint=job['release_checkpoint'],original_release_checkpoint_sha256=digest(job['release_checkpoint']))
    (target/'full_native_model_replay.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();config=read(args.config)
    for item in config['source_receipts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen full released-model replay source changed')
    if args.worker:return worker(config,args.config)
    api=original.resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('The full released-model replay is already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):
        state.update(values);(base/'status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),
                MKL_NUM_THREADS=str(config['cpu_threads']),OPENBLAS_NUM_THREADS=str(config['cpu_threads']))
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen([config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.replay_crossad_full_release_model_v1','--config',args.config,'--worker'],
                    cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid);update(dict(active_pid=child.pid,active_create_time=handle.create_time()))
                code,usage=original.capped_wait(child,config,api,update)
        target=ROOT/config['artifact_directory']/'full_native_model_replay.json'
        passed=code==0 and target.exists() and read(target).get('fixed_release_checkpoint_evaluation') is True
        receipt=dict(exit_code=code,full_native_model_replay_passed=passed,completed_training_seed_slots=0,**usage)
        (base/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,**receipt))
        if not passed:raise RuntimeError('Full released-model replay failed; preserve the whole attempt')
    finally:owner.close()


if __name__=='__main__':main()
