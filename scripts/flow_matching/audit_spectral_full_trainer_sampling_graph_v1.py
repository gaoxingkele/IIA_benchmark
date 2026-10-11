"""Compare native full two-cycle Trainer sampling under an inference graph repair.

The full upstream native derivative audit must actually succeed first. Both
native and reviewed trajectories use the same audit-only CPU RNG state and
complete original batch/grid/EMA model; no training or benchmark scores arise.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_spectral_full_sampling_derivative_graph_v1 import binding
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write,capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api

def registration(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full Trainer sampling source changed: '+pin['path'])
    upstream=read(ROOT/config['operator_config'])
    job,model=binding(upstream)
    if (config['canonical_id']!=job['id'] or config['full_generated_shape']!=[4002,24,6]
            or config['original_reference_samples']!=3662 or config['full_generation_cycles']!=2
            or config['full_batch_shape']!=[2001,24,6] or config['time_grid_points']!=500
            or config['expected_native_field_calls_per_cycle']!=500 or config['full_field_calls_per_pass']!=1000
            or config['solver_method']!='midpoint' or config['step_size']!=2./500
            or config['cpu_threads']!=upstream['cpu_threads']):
        raise ValueError('Full two-cycle original4002window/500gridpoint sampling recipe required')
    return upstream,job,model

def upstream_contract(receipt,proof):
    if (receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('full_native_derivative_graph_audit_passed') is not True
            or proof.get('passed') is not True or proof.get('actual_model_parameters')!=356800
            or proof.get('full_batch_shape')!=[2001,24,6] or proof.get('full_cases')!=5
            or proof.get('all_full_first_derivatives_bitwise_equal') is not True
            or proof.get('complete_native_EMA_checkpoint_restored_bitwise') is not True
            or proof.get('native_EMA_detached_parameter_flags_restored') is not True
            or proof.get('original_training_higher_order_forward_restored') is not True):
        raise ValueError('Actual complete native full-batch operator success required')
    cases=proof.get('cases',[])
    if len(cases)!=5 or [r.get('time') for r in cases]!=[0.,.25,.5,.75,.99999]:
        raise ValueError('All five original full native operator cases required')
    for case in cases:
        if (case.get('shape')!=[2001,24,6] or case.get('full_field_elements')!=288144
                or case.get('bitwise_derivatives_equal') is not True
                or case.get('Torch_RNG_consumption_unchanged') is not True
                or case.get('all_full_native_parameters_unchanged') is not True):
            raise ValueError('Every full native operator array/gradient/RNG case must verify')

def verify_upstream(config,upstream,arrays=False):
    receipt=read(ROOT/config['operator_receipt'])
    proof=read(ROOT/config['operator_proof'])
    upstream_contract(receipt,proof)
    if proof['canonical_id']!=config['canonical_id'] or proof['checkpoint_sha256']!=sha(upstream['checkpoint']):
        raise ValueError('Same complete originally trained model checkpoint required')
    for case in proof['cases']:
        if sha(case['arrays'])!=case['arrays_sha256']:
            raise ValueError('Frozen upstream complete operator arrays changed')
        if arrays:
            import numpy as np
            with np.load(case['arrays'],allow_pickle=False) as saved:
                if (saved['original'].shape!=(2001,24,6) or saved['reviewed'].shape!=(2001,24,6)
                        or saved['initial'].shape!=(2001,24,6) or saved['time'].shape!=(2001,1)
                        or not np.isfinite(saved['original']).all()
                        or not np.array_equal(saved['original'],saved['reviewed'])):
                    raise ValueError('All actual full upstream native derivative arrays must agree')
    return proof

def full_trajectory_contract(config,cycles,calls,shape):
    if (cycles!=config['full_generation_cycles'] or calls!=config['full_field_calls_per_pass']
            or list(shape)!=config['full_generated_shape']):
        raise ValueError('All original full sampling cycles, fields and windows must finish')

def bitwise_array_equal(left,right):
    return (left.shape==right.shape and left.dtype==right.dtype
            and left.tobytes(order='C')==right.tobytes(order='C'))

def worker(config):
    upstream,job,model_config=registration(config)
    operator_proof=verify_upstream(config,upstream,arrays=True)
    import numpy as np
    import torch
    import yaml
    from types import SimpleNamespace,MethodType
    from torch.utils.data import DataLoader
    from iia_benchmark.models.fm_spectral_author import build_author_model,FrozenAuthorGenerationDataset
    from iia_benchmark.models.fm_spectral_sampling_graph_review_v1 import sampling_first_derivative_graph
    params=model_config['parameters']
    torch.set_num_threads(config['cpu_threads'])
    model=build_author_model(ROOT/params['source_root'],params['model_target'],params['parameters'])
    from engine.solver import Trainer
    from flow_matching.solver import ODESolver
    target=ROOT/config['output_directory']
    artifact=Path(config['artifact_directory'])
    if target.exists() or artifact.exists():
        raise FileExistsError('Preserve complete and partial full native integrated audits')
    target.mkdir(parents=True)
    artifact.mkdir(parents=True)
    author_yaml=yaml.safe_load((ROOT/job['author_yaml']).read_text(encoding='utf-8'))
    data=FrozenAuthorGenerationDataset(ROOT/job['frozen_samples'])
    if len(data)!=3662 or sha(ROOT/job['frozen_samples'])!=job['data_sha256']:
        raise ValueError('All original frozen training data must remain bound')
    loader=DataLoader(data,batch_size=author_yaml['dataloader']['batch_size'],shuffle=True,
                      num_workers=0,pin_memory=True,drop_last=True)
    args=SimpleNamespace(save_dir=str(artifact/'native_author_output'))
    trainer=Trainer(author_yaml,args,model,{'dataloader':loader,'dataset':data},logger=None)
    copied=Path(args.save_dir)/'checkpoints/checkpoint-10.pt'
    shutil.copyfile(upstream['checkpoint'],copied)
    if sha(copied)!=sha(upstream['checkpoint']):
        raise ValueError('Copied complete original sampling checkpoint differs')
    trainer.load(10)
    saved=torch.load(copied,map_location='cpu',weights_only=True)
    if trainer.step!=10000 or sum(p.numel() for p in trainer.model.parameters())!=356800:
        raise ValueError('Full native author trained model must restore')
    ema=trainer.ema.ema_model
    expected={k[len('ema_model.'):]:v for k,v in saved['ema'].items() if k.startswith('ema_model.')}
    if (any(p.requires_grad for p in ema.parameters())
            or any(not torch.equal(v,expected[k]) for k,v in ema.state_dict().items())
            or any(not torch.equal(v,saved['model'][k]) for k,v in trainer.model.state_dict().items())):
        raise ValueError('Actual full native Trainer model and detached EMA weights must restore bitwise')
    count=config['full_field_calls_per_pass']
    fullshape=tuple(config['full_batch_shape'])
    input_path,field_path=artifact/'native_all_field_inputs.npy',artifact/'native_all_field_gradients.npy'
    inputs=np.lib.format.open_memmap(input_path,mode='w+',dtype='float32',shape=(count,)+fullshape)
    fields=np.lib.format.open_memmap(field_path,mode='w+',dtype='float32',shape=(count,)+fullshape)
    original_records=[]
    original_source_forward=ema.net.forward.__func__
    original_ode_sample=ODESolver.sample
    original_compute_flow=ema.compute_flow
    if 'compute_flow' in ema.__dict__:
        raise ValueError('Unmodified original full native flow entry point required')
    @contextmanager
    def trace(reviewed):
        cursor={'cycles':0,'calls':0}
        def observed_flow(instance,x,t):
            index=cursor['calls']
            if index>=count or tuple(x.shape)!=fullshape or tuple(t.shape)!=(2001,1):
                raise ValueError('Every native solver call must keep the full original batch')
            grad,aux=original_compute_flow(x,t)
            x_values=x.detach().cpu().numpy()
            grad_values=grad.detach().cpu().numpy()
            if (tuple(grad.shape)!=fullshape or grad.dtype!=torch.float32 or not np.isfinite(grad_values).all()
                    or not np.isfinite(x_values).all() or not all(math.isfinite(v) for v in aux.values())):
                raise ValueError('Every full native field must remain finite and complete')
            row=dict(call=index,time=float(t[0,0]),auxiliary=aux)
            if reviewed:
                if (grad.requires_grad or grad.grad_fn is not None or row!=original_records[index]
                        or not bitwise_array_equal(x_values,inputs[index]) or not bitwise_array_equal(grad_values,fields[index])):
                    raise ValueError('Every complete integrated field/input/time/auxiliary value must agree bitwise')
            else:
                if not grad.requires_grad:
                    raise ValueError('Observe the actual original higher-order graph before repair')
                inputs[index]=x_values
                fields[index]=grad_values
                original_records.append(row)
            cursor['calls']+=1
            if cursor['calls']%10==0:
                write(target/'progress.json',dict(phase='reviewed_full_native_Trainer_sampling' if reviewed else 'original_full_native_Trainer_sampling',
                    full_field_calls=cursor['calls'],required_field_calls=count,full_original_batch_shape=list(fullshape),
                    complete_native_cycles=cursor['cycles']))
            return grad,aux
        def observed_sample(instance,*a,**kw):
            # Preserve and forward the exact native ODE call, including its grid.
            grid=kw.get('time_grid')
            wanted=torch.linspace(0.,1.-1e-5,500,device='cpu')
            if (a or kw.get('method')!='midpoint' or kw.get('step_size')!=2./500
                    or grid is None or not torch.equal(grid,wanted)
                    or tuple(kw['x_init'].shape)!=fullshape):
                raise ValueError('Native complete500point midpoint ODE call required')
            before=cursor['calls']
            result=original_ode_sample(instance,*a,**kw)
            if cursor['calls']-before!=500:
                raise ValueError('All500 native midpoint vector-field evaluations required per cycle')
            cursor['cycles']+=1
            return result
        ema.compute_flow=MethodType(observed_flow,ema)
        ODESolver.sample=observed_sample
        try:
            yield cursor
        finally:
            del ema.compute_flow
            ODESolver.sample=original_ode_sample
    torch.manual_seed(config['audit_sampling_seed'])
    sampling_rng=torch.get_rng_state().clone()
    start=time.perf_counter()
    with torch.no_grad(),trace(False) as native_count:
        native=trainer.sample(num=len(data),size_every=2001,shape=[data.window,data.var_num])
    original_seconds=time.perf_counter()-start
    full_trajectory_contract(config,native_count['cycles'],native_count['calls'],native.shape)
    original_rng_after=torch.get_rng_state().clone()
    inputs.flush()
    fields.flush()
    write(target/'native_all_solver_call_metadata.json',dict(full_field_calls=count,records=original_records))
    np.save(artifact/'native_full_generated.npy',native,allow_pickle=False)
    torch.set_rng_state(sampling_rng)
    start=time.perf_counter()
    with torch.no_grad(),sampling_first_derivative_graph(ema) as patch,trace(True) as reviewed_count:
        reviewed=trainer.sample(num=len(data),size_every=2001,shape=[data.window,data.var_num])
    reviewed_seconds=time.perf_counter()-start
    full_trajectory_contract(config,reviewed_count['cycles'],reviewed_count['calls'],reviewed.shape)
    if (not np.isfinite(native).all() or not np.isfinite(reviewed).all() or not bitwise_array_equal(native,reviewed)
            or not torch.equal(original_rng_after,torch.get_rng_state())
            or ema.net.forward.__func__ is not original_source_forward
            or any(not torch.equal(v,expected[k]) for k,v in ema.state_dict().items())
            or torch.cuda.is_initialized()):
        raise ValueError('All full generated arrays, final RNG and native EMA/training forward must match')
    np.save(artifact/'reviewed_full_generated.npy',reviewed,allow_pickle=False)
    np.savez_compressed(artifact/'audit_sampling_rng.npz',before=sampling_rng.numpy(),after=original_rng_after.numpy())
    del inputs,fields
    proof=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,canonical_id=job['id'],
        checkpoint_sha256=sha(copied),full_native_Trainer_sample_executed_twice=True,
        original_generated_shape=list(native.shape),reviewed_generated_shape=list(reviewed.shape),
        full_original_generation_cycles=2,complete_field_calls_per_pass=1000,total_actual_native_and_reviewed_field_calls=2000,
        original_time_grid_points_per_cycle=500,original_midpoint_step_size=2./500,
        every_full_field_input_derivative_time_and_auxiliary_bitwise_equal=True,all4002_generated_windows_bitwise_equal=True,
        all_full_native_Trained_and_EMA_weights_restored_bitwise=True,final_Torch_CPU_RNG_state_equal=True,
        original_training_higher_order_forward_restored=True,original_seconds=original_seconds,reviewed_seconds=reviewed_seconds,
        sampling_patch=patch,upstream_full_native_operator_proof_sha256=sha(ROOT/config['operator_proof']),
        upstream_full_native_operator_receipt_sha256=sha(ROOT/config['operator_receipt']),
        artifact_receipts=[dict(path=str(p),sha256=sha(p)) for p in sorted(artifact.iterdir()) if p.is_file()],
        cuda_context_initialized=False,full_CPU_integrated_sampling_equivalence_verified=True,
        CUDA_integrated_sampling_equivalence_certified=False,original_post_training_sampling_RNG_resume_certified=False,
        original_training_updates=10000,new_training_runs=0,additional_training_seed_slots=0,benchmark_performance=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary='Actual full original native Trainer.sample(num3662,size_every2001), two500gridpoint midpoint cycles, full4002x24x6 output and all1000complete field inputs/gradients/auxiliary values compared bitwise per pass. Complete originally trained10000step356800parameter EMA checkpoint; fixed audit-only CPU sampling RNG. This verifies the inference graph candidate only on this full native CPU trajectory, not original unsaved post-training CUDA RNG, new training, learned evaluators or benchmark performance. Full original fresh training/sampling/four five-repeat evaluators remain required.')
    write(target/'full_native_Trainer_sampling_graph_audit.json',proof)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/args.config)
    upstream,job,_=registration(config)
    if args.probe_only:
        print(json.dumps(dict(full_generated_shape=config['full_generated_shape'],field_calls_per_pass=1000,
            canonical_id=job['id'],upstream_complete=(ROOT/config['operator_receipt']).exists(),
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,benchmark_performance=False)))
        return
    if args.worker:
        worker(config)
        return
    settings=config['settings']
    if settings['minimum_commit_headroom_bytes']<settings['emergency_commit_headroom_bytes']+settings['maximum_worker_private_bytes']+2*1024**3:
        raise ValueError('Full original integration memory plus emergency reserve required')
    api=resource_api()
    base=ROOT/config['state_root']
    base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('Full native Trainer sampling audit controller already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        configuration_sha256=sha(ROOT/args.config),state='waiting_actual_full_native_operator_success')
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json',state)
    try:
        while not (ROOT/config['operator_receipt']).exists():
            try:
                p=psutil.Process(config['expected_operator_controller_pid'])
                live=(abs(p.create_time()-config['expected_operator_controller_create_time'])<.001
                    and config['operator_config'] in p.cmdline())
            except psutil.NoSuchProcess:
                live=False
            except (psutil.AccessDenied,OSError) as error:
                update(dict(state='waiting_operator_observation_retry',observation_error=type(error).__name__))
                time.sleep(10)
                continue
            update(dict(state='waiting_actual_full_native_operator_success',actual_operator_identity_live=live))
            if not live:
                # Recheck actual terminal evidence; never restart on observation expiry.
                time.sleep(10)
                if (ROOT/config['operator_receipt']).exists():break
                raise RuntimeError('Exact upstream operator identity absent with no terminal receipt')
            time.sleep(10)
        verify_upstream(config,upstream)
        with api['resource_slot'](ROOT/config['resource_lock'],settings,update):
            command=[str(ROOT/config['python']),'-X','utf8','-u','-m',
                'scripts.flow_matching.audit_spectral_full_trainer_sampling_graph_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',
                        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,config,api,update)
        path=ROOT/config['output_directory']/'full_native_Trainer_sampling_graph_audit.json'
        passed=code==0 and path.exists() and read(path)['passed']
        write(base/'receipt.json',dict(exit_code=code,full_native_Trainer_sampling_graph_audit_passed=passed,**usage))
        update(dict(active_pid=None,state='completed' if passed else 'failed_preserved'))
        if not passed:
            raise RuntimeError('Preserve full native integration failure; no shortened trajectory substitute')
    except BaseException as error:
        update(dict(state='failed_preserved',error=repr(error)))
        raise
    finally:
        owner.close()

if __name__=='__main__':
    main()
