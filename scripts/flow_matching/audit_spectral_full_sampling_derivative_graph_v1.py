"""Compare complete native 2001x24x6 fields, preserving the first derivative.

Full trained native architecture/EMA weights are required. This operator audit
does not certify an integrated trajectory, perform training, or supply a score.
"""
import argparse
from datetime import datetime,timezone
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write,capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def binding(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen full native sampling audit changed: '+item['path'])
    queue=read(ROOT/config['queue'])
    job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    model=read(ROOT/job['model_config'])
    if (len(queue['jobs'])!=70 or job['model']!='spectral_flow' or job['dataset']!='stocks'
            or job['training_updates']!=10000 or job['generation_batch_size']!=2001
            or job['generation_field_chunk_size'] is not None or job['data_shape']!=[3662,24,6]
            or model['parameters']['parameters']!=dict(seq_length=24,feature_size=6,n_layers=10,d_model=64,
                sampling_timesteps=500,mlp_hidden_times=1,sigma=1e-5,solver_method='midpoint')
            or config['full_batch_shape']!=[2001,24,6] or config['case_times']!=[0.0,.25,.5,.75,.99999]):
        raise ValueError('Complete original batch/architecture/500-grid midpoint recipe required')
    return job,model


def worker(config):
    job,registration=binding(config)
    import numpy as np
    import torch
    from ema_pytorch import EMA
    from iia_benchmark.models.fm_spectral_author import build_author_model
    from iia_benchmark.models.fm_spectral_sampling_graph_review_v1 import sampling_first_derivative_graph
    torch.set_num_threads(config['cpu_threads'])
    parameters=registration['parameters']
    model=build_author_model(ROOT/parameters['source_root'],parameters['model_target'],parameters['parameters'])
    if sum(p.numel() for p in model.parameters())!=356800:
        raise ValueError('Full original native trained architecture required')
    checkpoint=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint['step']!=10000:
        raise ValueError('Complete original training checkpoint required')
    state={k[len('ema_model.'):]:v for k,v in checkpoint['ema'].items() if k.startswith('ema_model.')}
    # Instantiate the actual native EMA implementation so parameter detach flags
    # match Trainer's original sampling model as well as its tensor values.
    ema=EMA(model,beta=.995,update_every=10)
    ema.load_state_dict(checkpoint['ema'],strict=True)
    model=ema.ema_model
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Actual native detached EMA parameter flags required')
    if any(not torch.equal(value,state[key]) for key,value in model.state_dict().items()):
        raise ValueError('All full native EMA weights must restore bitwise')
    target=ROOT/config['output_directory']
    artifact=Path(config['artifact_directory'])
    if target.exists() or artifact.exists():
        raise FileExistsError('Preserve complete or partial native operator audit')
    target.mkdir(parents=True)
    artifact.mkdir(parents=True)
    generator=torch.Generator(device='cpu').manual_seed(config['audit_input_seed'])
    initial=torch.randn(config['full_batch_shape'],generator=generator)
    cases=[]
    for index,t in enumerate(config['case_times']):
        full_time=torch.full([len(initial),1],t)
        RNG_before=torch.get_rng_state().clone()
        write(target/'progress.json',dict(phase='native_original_full_batch_field',case=index,
            required_cases=len(config['case_times']),full_batch_shape=config['full_batch_shape']))
        start=time.perf_counter()
        with torch.no_grad(): native,old_aux=model.compute_flow(initial,full_time)
        if not native.requires_grad:
            raise ValueError('Original native higher-order derivative graph must be observed')
        original=native.detach().clone()
        del native
        gc.collect()
        original_seconds=time.perf_counter()-start
        write(target/'progress.json',dict(phase='reviewed_full_batch_first_derivative',case=index,
            required_cases=len(config['case_times']),full_batch_shape=config['full_batch_shape']))
        start=time.perf_counter()
        with torch.no_grad(),sampling_first_derivative_graph(model) as patch:
            reviewed,new_aux=model.compute_flow(initial,full_time)
        if reviewed.requires_grad or reviewed.grad_fn is not None:
            raise ValueError('Sampling must retain the full first derivative but no higher-order graph')
        if (original.shape!=initial.shape or reviewed.shape!=initial.shape or not torch.isfinite(original).all()
                or not torch.isfinite(reviewed).all() or not torch.equal(original,reviewed)
                or old_aux!=new_aux or not torch.equal(RNG_before,torch.get_rng_state())):
            raise ValueError('All full native derivatives/auxiliary values/RNG must agree bitwise')
        if any(not torch.equal(value,state[key]) for key,value in model.state_dict().items()):
            raise ValueError('Native parameter/buffer state must remain bitwise unchanged')
        path=artifact/('full_case_%02d.npz'%index)
        np.savez_compressed(path,original=original.numpy(),reviewed=reviewed.numpy(),initial=initial.numpy(),
            time=full_time.numpy())
        cases.append(dict(case=index,time=t,shape=list(original.shape),full_field_elements=original.numel(),
            bitwise_derivatives_equal=True,auxiliary_values_equal=True,Torch_RNG_consumption_unchanged=True,
            original_higher_order_graph_observed=True,reviewed_higher_order_graph_retained=False,
            all_full_native_parameters_unchanged=True,original_seconds=original_seconds,
            reviewed_seconds=time.perf_counter()-start,arrays=str(path),arrays_sha256=sha(path),patch=patch))
        del original,reviewed
        gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError('Full native operator audit must remain CPU-only')
    proof=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,canonical_id=job['id'],
        complete_native_EMA_checkpoint_restored_bitwise=True,native_EMA_detached_parameter_flags_restored=True,
        checkpoint_sha256=sha(config['checkpoint']),
        actual_model_parameters=356800,full_batch_shape=config['full_batch_shape'],cases=cases,
        full_cases=len(cases),all_full_first_derivatives_bitwise_equal=True,
        original_training_higher_order_forward_restored=True,
        original_training_updates=10000,original_sampling_time_grid_points=500,
        original_midpoint_step_size=2.0/500,full_integrated_native_sampling_executed=False,
        full_integrated_sampling_equivalence_certified=False,benchmark_performance=False,
        cuda_context_initialized=False,additional_training_seed_slots=0,author_equivalence_certified=False,
        all_original_experiments_complete=False,
        boundary='Actual complete356800parameter trained EMA model and detached parameter flags; five original2001x24x6 full-vector fields compared bitwise, with unchanged auxiliary values/weights/Torch RNG and no chunking. Only inference-time higher-order derivative graph retention changes; native first derivative and original training derivative are retained. This does not run or certify a complete500-grid-point integrated sampling trajectory or any benchmark score. Full original training/sampling/four evaluators remain required for recovery.')
    write(target/'full_native_derivative_graph_audit.json',proof)
    print(dict(passed=True,full_original_field_cases=len(cases),integrated_sampling_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/args.config)
    job,model=binding(config)
    if args.probe_only:
        print(dict(canonical_id=job['id'],full_shape=config['full_batch_shape'],
            full_cases=len(config['case_times']),torch_imported='torch' in sys.modules,
            native_operator_audit_not_performance=True))
        return
    if args.worker:
        worker(config)
        return
    if config['settings']['minimum_commit_headroom_bytes']<config['settings']['emergency_commit_headroom_bytes']+config['settings']['maximum_worker_private_bytes']+2*1024**3:
        raise ValueError('Complete native operator memory plus shared reserve required')
    api=resource_api()
    base=ROOT/config['state_root']
    base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('Same native full-batch audit already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        state='starting',configuration_sha256=sha(ROOT/args.config))
    def update(values):
        state.update(values)
        state['captured_utc']=datetime.now(timezone.utc).isoformat()
        write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[str(ROOT/config['python']),'-X','utf8','-u','-m',
                'scripts.flow_matching.audit_spectral_full_sampling_derivative_graph_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),
                        MKL_NUM_THREADS=str(config['cpu_threads'])),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,config,api,update)
        path=ROOT/config['output_directory']/'full_native_derivative_graph_audit.json'
        passed=bool(code==0 and path.exists() and read(path)['passed'])
        write(base/'receipt.json',dict(exit_code=code,full_native_derivative_graph_audit_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None))
        if not passed:
            raise RuntimeError('Preserve original complete geometry and actual failed capacity evidence')
    finally:
        owner.close()


if __name__=='__main__':
    main()
