"""Isolate the actual CICIDS resource abort; preserve every full CPU forward.

Immutable per-forward reconstruction checkpoints permit exact interrupted
replay resumption. Original source/results and the all78 controller stay intact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import FunctionType, SimpleNamespace
import uuid

from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as original

ROOT=original.ROOT
MODULE='scripts.flow_matching.recover_moment_CICIDS_full_durable_v1'
CONFIG='configs/runtime/fm_moment_CICIDS_full_durable_recovery.v1.json'
PARENT='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
IDENTITY='fm_moment_pretrained_base__release512__cicids__seed42'
INCIDENT='experiments/runs/fm_moment_full78_checkpoint_replay_v1/attempts/'+IDENTITY+'/native/guard_receipt.json'
sha=original.sha
read=original.read
write=original.write


def registration():
    parent=read(ROOT/PARENT)
    settings=dict(parent['settings'],minimum_commit_headroom_bytes=40*2**30)
    paths=[PARENT,INCIDENT,MODULE.replace('.','/')+'.py','tests/test_fm_moment_CICIDS_full_durable_v1.py']
    return dict(original_runtime=PARENT,original_runtime_sha256=sha(ROOT/PARENT),original_job=IDENTITY,
        incident=INCIDENT,incident_sha256=sha(ROOT/INCIDENT),settings=settings,resource_lock=parent['resource_lock'],
        state_root='experiments/runs/fm_moment_CICIDS_full_durable_recovery_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_CICIDS_full_durable_recovery_v1',
        forward_cache_root='F:/aicoding/IIA_Data/experiments/moment_CICIDS_full_durable_recovery_v1/'+IDENTITY+'/full_forward_cache',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        original_scope=dict(parent['original_scope'],paper_scalar_targets=702),
        citation='https://arxiv.org/abs/2402.03885',
        source_model_config='configs/models/fm_moment_pretrained/fm_moment_pretrained_base__release512.json',
        native_CPU_environment_and_batch_and_all_windows_unchanged=True,
        cache='Immutable complete original forward reconstructions;full input/mask/checkpoint/source/environment hashes;no partial forward accepted',
        original78_queue_unchanged=True,additional_training_seed_slots=0,
        all_original_experiments_complete=False)


def contract(c):
    if c!=registration():raise ValueError('Isolated actual abort and original full numerical contract required')
    for item in c['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen recovery dependency changed')
    parent=read(ROOT/PARENT)
    cases=original.verify(parent)
    expected=dict(parent['settings'],minimum_commit_headroom_bytes=40*2**30)
    if c['settings']!=expected:raise ValueError('Only admission commit is strengthened;all other guards retained')
    incident=read(ROOT/INCIDENT)
    if incident['exit_code']!=1 or not incident['resource_guard_aborted'] or incident['passed']:
        raise ValueError('Actual original terminal resource abort required')
    import psutil
    handle=incident['exact_child_handle']
    try:
        p=psutil.Process(handle['pid'])
        if p.create_time()==handle['create_time']:raise ValueError('Original guarded worker still live')
    except psutil.NoSuchProcess:pass
    # The source runs under a Windows venv launcher; its known numerical child
    # must also be terminal, independently of the launcher identity.
    try:
        p=psutil.Process(25352)
        if p.create_time()==1792106881.7437527:raise ValueError('Original numerical worker still live')
    except psutil.NoSuchProcess:pass
    runtime=dict(parent,artifact_root=c['artifact_root'],_config_path=CONFIG)
    return runtime,cases[IDENTITY]


class FullForwardCache:
    def __init__(self,directory,binding):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.binding=binding
        self.calls=0;self.actual_new_forwards=0;self.resumed_forwards=0
        self.binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()

    def wrap(self,forward):
        def cached(*args,**kwargs):
            import numpy as np
            import torch
            if args or set(kwargs)!={'x_enc','input_mask','mask'} or not torch.is_inference_mode_enabled():
                raise ValueError('Only complete original inference-mode forward calls may be cached')
            arrays={}
            for name,value in kwargs.items():
                if value.device.type!='cpu' or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Preserve full original finite CPU FP32 inputs and masks')
                arrays[name]=value.detach().numpy()
            description={name:dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest()) for name,value in arrays.items()}
            key_data=dict(binding_sha256=self.binding_sha256,ordinal=self.calls,inputs=description)
            key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()
            target=self.directory/key
            if target.exists():
                receipt=read(target/'receipt.json')
                if (receipt['key_data']!=key_data or not receipt['full_original_forward_completed']
                        or receipt['training_updates']!=0 or receipt['array_sha256']!=sha(target/'reconstruction.npy')):
                    raise ValueError('Durable full-forward source/input/array identity changed')
                reconstruction=np.load(target/'reconstruction.npy',allow_pickle=False)
                if (list(reconstruction.shape)!=description['x_enc']['shape'] or reconstruction.dtype!=np.float32
                        or not np.isfinite(reconstruction).all()):
                    raise ValueError('Every complete cached reconstruction coordinate required')
                output=SimpleNamespace(reconstruction=torch.from_numpy(reconstruction))
                self.resumed_forwards+=1
            else:
                output=forward(**kwargs)
                value=output.reconstruction
                if (value.shape!=kwargs['x_enc'].shape or value.device.type!='cpu'
                        or value.dtype!=torch.float32 or not torch.isfinite(value).all()):
                    raise ValueError('Original complete forward did not produce finite CPU FP32 reconstruction')
                # Commit a whole directory atomically. Interrupted pending dirs
                # remain preserved and are never accepted as completed cache.
                pending=self.directory/('pending-'+uuid.uuid4().hex)
                pending.mkdir(exist_ok=False)
                with (pending/'reconstruction.npy').open('xb') as stream:
                    np.save(stream,value.detach().numpy(),allow_pickle=False)
                    stream.flush();os.fsync(stream.fileno())
                receipt=dict(key_data=key_data,array_sha256=sha(pending/'reconstruction.npy'),
                    full_original_forward_completed=True,training_updates=0)
                with (pending/'receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
                    json.dump(receipt,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
                pending.rename(target)
                self.actual_new_forwards+=1
            self.calls+=1
            return output
        return cached

    def summary(self):
        return dict(full_forward_calls=self.calls,actual_new_forwards=self.actual_new_forwards,
            resumed_original_full_forwards=self.resumed_forwards,binding_sha256=self.binding_sha256,
            all_reconstruction_coordinates_retained=True,full_original_forward_order_preserved=True,
            fresh_checkpoint_inference_in_every_attempt=self.actual_new_forwards>0,
            forward_budget_or_precision_reduced=False)


def worker(c,phase):
    runtime,case=contract(c)
    if phase=='independent':
        original.review_worker(runtime,case)
        return
    result,_,binding=original.check_source(runtime,case)
    environment=original.environment_matches(result)
    cache=FullForwardCache(c['forward_cache_root'],
        dict(source_binding=binding,environment=environment,numerical_source_runtime_sha256=c['original_runtime_sha256'],cache_protocol_version=1,
             model_config=case['config'],window=case['job']['window'],batch_size=1,CPU_threads=2))
    def observe(model,window,tolerance):
        if model.model.training:raise ValueError('Durable replay requires the original frozen eval model')
        model.model.forward=cache.wrap(model.model.forward)
        return original.observed_score(model,window,tolerance)
    def recorded(path,data):
        if Path(path).name=='native_replay.json':
            if cache.calls!=data['inline_full_window_MSE_check']['forward_calls']:
                raise ValueError('Every original forward and independent MSE check must be accounted')
            data=dict(data,durable_full_forward_resumption=cache.summary(),
                original_resource_abort_preserved=True,original_78_controller_modified=False)
        write(path,data)
    native=FunctionType(original.native_worker.__code__,dict(original.native_worker.__globals__,
        observed_score=observe,write=recorded),argdefs=original.native_worker.__defaults__)
    native(runtime,case)


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    runtime,case=contract(c);api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    lock=api['exclusive_lock'](base/'controller.lock')
    if lock is None or (base/'status.json').exists():
        if lock:lock.close()
        raise FileExistsError('Preserve this isolated full recovery attempt;do not restart it')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        original_job=IDENTITY,full_models_verified=0,original_78_queue_unchanged=True,all_original_experiments_complete=False)
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        for phase in ['native','independent']:
            with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
                command=[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',MODULE,'--phase',phase]
                with (base/(phase+'.stdout.log')).open('xb') as out,(base/(phase+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,
                        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')))
                    handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                    update(dict(state='running_full_'+phase,active_pid=child.pid,exact_child_handle=handle))
                    code,usage=capped_wait(child,dict(runtime,settings=c['settings']),api,update)
                proof=Path(c['artifact_root'])/IDENTITY/('native/native_replay.json' if phase=='native' else 'independent/review.json')
                passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                write(base/(phase+'.guard.json'),dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
                    proof_path=str(proof),proof_sha256=sha(proof) if proof.exists() else None))
                update(dict(active_pid=None))
                if not passed:raise RuntimeError('Preserved isolated recovery requires another version;do not retry blindly')
        update(dict(state='complete_full_native_and_independent_review_other_original_models_pending',full_models_verified=1))
    except Exception as error:
        update(dict(state='failed_preserved_requires_isolated_repair',error=repr(error)));raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--phase',choices=['native','independent']);args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve the registered recovery')
        write(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG)
    if args.phase:worker(c,args.phase)
    else:controller(c)


if __name__=='__main__':main()
