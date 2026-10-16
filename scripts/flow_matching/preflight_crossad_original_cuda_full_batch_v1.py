"""Verify original CrossAD CUDA at full native shape without training updates.

Original CPU workers and immutable recipes remain untouched. This is a new
explicit device protocol, never evidence of completed benchmark training.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import FunctionType
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.preflight_crossad_original_cuda_full_batch_v1'
CONFIG='configs/reproducibility/fm_crossad_original_cuda_full_batch_preflight.v1.json'
QUEUE='configs/experiments/fm_crossad_complete_queue.v7.json'
MODEL='configs/models/fm_crossad_original_cuda_fp32.v1.json'


def registration():
    q=read(ROOT/QUEUE);settings=q['settings']
    job=next(j for j in q['jobs'] if j['id']=='crossad_author__fresh_train__gecco__seed1105')
    for item in job['frozen_files']:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Original registered CrossAD source/data changed: '+item['path'])
    old=read(ROOT/'docs/reports/fm_crossad_complete_preflight_v4_GECCO_2026-10-09.json')
    shape=old['records'][0]['native_batch_shape']
    if shape!=[job['train_parameters']['batch_size'],job['model_parameters']['seq_len'],9]:
        raise ValueError('Same actual original128x128x9 batch required')
    files=[MODULE.replace('.','/')+'.py',MODEL,QUEUE,
        'tests/test_fm_crossad_original_cuda_full_batch_v1.py',
        'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_crossad_author_job.py',
        'src/iia_benchmark/models/fm_crossad_author_runtime.py',
        'docs/reports/fm_crossad_complete_preflight_v4_GECCO_2026-10-09.json']
    files += [p['path'] for p in job['frozen_files']]
    # The original CUDA source retains its two explicit .cuda() mask placements.
    source=ROOT/'experiments/runs/mtsad_protocol_audit/sources/crossad/original'
    files += [p.relative_to(ROOT).as_posix() for p in source.rglob('*.py') if '__pycache__' not in p.parts]
    return dict(job=job,source_queue=QUEUE,source_queue_sha256=sha(ROOT/QUEUE),
        original_source_root=source.relative_to(ROOT).as_posix(),cpu_reference_root=settings['runtime_root'],
        raw_root=settings['raw_root'],cache_root=settings['cache_root'],python=settings['python'],
        full_native_batch_shape=shape,all_source_training_windows=old['records'][0]['training_windows_stride1'],
        parameters=old['records'][0]['parameter_count'],cpu_threads=settings['cpu_threads'],
        settings={k:settings[k] for k in ['minimum_available_memory_bytes','minimum_commit_headroom_bytes','emergency_commit_headroom_bytes']},
        resource_lock='experiments/runs/fm_original_cuda0_exclusive.lock',
        GPU=dict(device=0,minimum_free_bytes=18*2**30,emergency_free_bytes=2*2**30,
            allocator_fraction=.75,maximum_allocated_bytes=18*2**30),
        fp32_precision='ieee',autocast=False,TF32=False,
        deterministic_algorithms_warn_only=True,bitwise_training_determinism_certified=False,
        CPU_GPU_array_tolerance=dict(rtol=2e-6,atol=2e-6),optimizer_updates=0,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_crossad_original_cuda_full_batch_preflight_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_original_cuda_full_batch_preflight_v1',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        benchmark_performance=False,all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Full original shape,source,precision and unchanged host gates required')


def gpu_snapshot():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=result.stdout.strip().splitlines()[0].split(',')
    return dict(index=int(row[0]),name=row[1].strip(),total_bytes=int(row[2])*2**20,
                free_bytes=int(row[3])*2**20,utilization_percent=int(row[4]))


def guarded_GPU_wait(child,c,update):
    from scripts.flow_matching import heavy_resources as original
    violations=[]
    def snapshot():
        memory=original.resource_snapshot()
        try:
            GPU=gpu_snapshot()
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            # A transient observation failure is not an experiment failure.
            update(dict(GPU_observation_error=repr(error)))
            return memory
        update(dict(GPU=GPU))
        if GPU['free_bytes']<c['GPU']['emergency_free_bytes']:
            violations.append(GPU)
            original.terminate_tree(child)
        return memory
    # Retain the complete original CPU memory formula, checks and polling loop.
    wait=FunctionType(original.guarded_wait.__code__,dict(original.guarded_wait.__globals__,resource_snapshot=snapshot),
        argdefs=original.guarded_wait.__defaults__)
    code,usage=wait(child,c['settings'],update)
    usage['GPU_resource_guard_aborted']=bool(violations)
    usage['resource_guard_aborted']=usage['resource_guard_aborted'] or bool(violations)
    if violations:usage['GPU_abort_snapshot']=violations[0]
    return code,usage


def cuda_worker(c):
    import numpy as np
    import torch
    from iia_benchmark.models.fm_crossad_author_runtime import build_model,install_wide_cache
    if not torch.cuda.is_available():raise ValueError('Actual CUDA device required; never silently run this probe on CPU')
    torch.set_num_threads(c['cpu_threads'])
    random.seed(c['job']['seed']);np.random.seed(c['job']['seed']);torch.manual_seed(c['job']['seed']);torch.cuda.manual_seed_all(c['job']['seed'])
    torch.backends.fp32_precision='ieee'
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.cuda.set_per_process_memory_fraction(c['GPU']['allocator_fraction'],0)
    torch.cuda.reset_peak_memory_stats()
    runtime=ROOT/c['original_source_root'];audits=install_wide_cache(runtime,ROOT/c['cache_root'])
    provider=__import__('importlib').import_module('data_provider.data_provider')
    data,loader=provider.data_provider(str(ROOT/c['raw_root']),c['job']['dataset'],128,128,1,'train')
    if len(data)!=c['all_source_training_windows']:raise ValueError('All original source training windows must remain eligible')
    values=np.stack([data.train[i:i+128] for i in range(128)]).astype(np.float32)
    if list(values.shape)!=c['full_native_batch_shape']:raise ValueError('Never shrink the real original batch,sequence,or channels')
    model=build_model(runtime,c['job']['model_parameters']).float().to('cuda:0')
    if sum(p.numel() for p in model.parameters())!=c['parameters']:raise ValueError('Complete original model required')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('All original weights must remain FP32')
    out=Path(c['artifact_root']);out.mkdir(parents=True,exist_ok=False)
    np.save(out/'full_native_input.npy',values,allow_pickle=False)
    # Retain the pristine initialization for exact-state cross-device inference.
    torch.save(model.state_dict(),out/'original_initialization.pth')
    model.eval();x=torch.from_numpy(values).to('cuda:0')
    start=time.monotonic()
    with torch.no_grad():
        first,_=model.infer(x,None,None,None)
        second,_=model.infer(x,None,None,None)
    torch.cuda.synchronize()
    if first.shape!=x.shape or not torch.isfinite(first).all() or not torch.equal(first,second):
        raise ValueError('Full original CUDA inference must be finite and repeat identically')
    np.save(out/'full_original_CUDA_scores.npy',first.cpu().numpy(),allow_pickle=False)
    inference_seconds=time.monotonic()-start
    del first,second
    model.train();model.zero_grad(set_to_none=True)
    start=time.monotonic();loss,_=model(x,None,None,None)
    if not torch.isfinite(loss):raise ValueError('Actual full source CUDA loss must be finite')
    loss.backward();torch.cuda.synchronize()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() and g.dtype==torch.float32 for g in gradients):
        raise ValueError('Actual original fullbatch CUDA backward must remain finite FP32')
    peak=torch.cuda.max_memory_allocated()
    if peak>c['GPU']['maximum_allocated_bytes']:raise ValueError('Registered GPU allocation cap exceeded')
    exclusive(out/'CUDA_report.json',dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),
        full_native_batch_shape=list(values.shape),full_source_training_windows=len(data),parameters=c['parameters'],
        full_batch_inference_repeat_bitwise_equal=True,all_scores_finite=True,finite_original_backward=True,
        gradient_tensors=len(gradients),optimizer_updates=0,torch_version=torch.__version__,CUDA_build=torch.version.cuda,
        device=torch.cuda.get_device_name(0),peak_GPU_allocated_bytes=peak,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
        precision=dict(global_fp32=torch.backends.fp32_precision,CUDA_matmul=torch.backends.cuda.matmul.fp32_precision,
            cuDNN=torch.backends.cudnn.fp32_precision,autocast_enabled=torch.is_autocast_enabled('cuda')),
        deterministic_warn_only=True,strict_bitwise_training_determinism_certified=False,
        full_batch_two_inference_seconds=inference_seconds,full_batch_backward_seconds=time.monotonic()-start,
        complete_input_sha256=sha(out/'full_native_input.npy'),initialization_sha256=sha(out/'original_initialization.pth'),
        complete_CUDA_scores_sha256=sha(out/'full_original_CUDA_scores.npy'),cache_audits=audits,
        native_probe_only_not_benchmark=True,all_original_experiments_complete=False))


def cpu_reference(c):
    import numpy as np
    import torch
    from iia_benchmark.models.fm_crossad_author_runtime import build_model
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['job']['seed'])
    out=Path(c['artifact_root']);gpu=read(out/'CUDA_report.json')
    if not gpu['passed'] or gpu['configuration_sha256']!=sha(ROOT/CONFIG):raise ValueError('Actual unchanged original fullbatch CUDA probe must pass first')
    model=build_model(ROOT/c['cpu_reference_root'],c['job']['model_parameters']).float()
    if sha(out/'original_initialization.pth')!=gpu['initialization_sha256']:raise ValueError('Original model initialization changed')
    model.load_state_dict(torch.load(out/'original_initialization.pth',map_location='cpu',weights_only=False),strict=True)
    values=np.load(out/'full_native_input.npy',allow_pickle=False)
    if sha(out/'full_native_input.npy')!=gpu['complete_input_sha256']:raise ValueError('Complete original native input changed')
    model.eval();start=time.monotonic()
    with torch.no_grad():scores,_=model.infer(torch.from_numpy(values),None,None,None)
    actual=scores.numpy();expected=np.load(out/'full_original_CUDA_scores.npy',allow_pickle=False)
    if actual.shape!=tuple(c['full_native_batch_shape']) or actual.dtype!=expected.dtype:
        raise ValueError('Every full original CPU/GPU score shape and dtype must match')
    maximum=float(np.max(np.abs(actual-expected)))
    within=bool(np.allclose(actual,expected,**c['CPU_GPU_array_tolerance']))
    np.save(out/'full_CPU_reference_scores.npy',actual,allow_pickle=False)
    report=dict(passed=within,configuration_sha256=sha(ROOT/CONFIG),all_native_score_entries_compared=int(actual.size),
        full_native_batch_shape=list(actual.shape),maximum_absolute_CPU_GPU_difference=maximum,
        fixed_tolerance=c['CPU_GPU_array_tolerance'],all_finite=bool(np.isfinite(actual).all()),
        CPU_reference_seconds=time.monotonic()-start,optimizer_updates=0,native_probe_only_not_benchmark=True,
        full_CPU_scores_sha256=sha(out/'full_CPU_reference_scores.npy'),CUDA_report_sha256=sha(out/'CUDA_report.json'),
        historical_framework_equivalence_certified=False,all_original_experiments_complete=False)
    exclusive(out/'CPU_GPU_report.json',report)
    if not within:raise ValueError('Full CPU/GPU scores differ beyond the fixed original replay tolerance; preserve without loosening')


def controller(c):
    import psutil
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.heavy_resources import guarded_wait
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=exclusive_lock(base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Preserve every original CUDA native preflight attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,optimizer_updates=0,
        benchmark_performance=False,all_original_experiments_complete=False)
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while gpu_snapshot()['free_bytes']<c['GPU']['minimum_free_bytes']:
            update(dict(state='waiting_registered_free_VRAM',GPU=gpu_snapshot()));time.sleep(10)
        with resource_slot(ROOT/c['resource_lock'],c['settings'],update):
            if gpu_snapshot()['free_bytes']<c['GPU']['minimum_free_bytes']:
                raise RuntimeError('VRAM changed before exclusive ownership; preserve attempt without launching')
            for phase in ['cuda','cpu_reference']:
                verify(c)
                command=[str(ROOT/c['python']),'-B','-X','utf8','-u','-m',MODULE,'--phase',phase]
                with (base/(phase+'.stdout.log')).open('xb') as out,(base/(phase+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                        PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='0' if phase=='cuda' else '-1',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS=str(c['cpu_threads']),MKL_NUM_THREADS=str(c['cpu_threads'])))
                    handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                    update(dict(state='running_full_original_'+phase,active_pid=child.pid,exact_child_handle=handle))
                    code,usage=guarded_GPU_wait(child,c,update) if phase=='cuda' else guarded_wait(child,c['settings'],update)
                path=Path(c['artifact_root'])/('CUDA_report.json' if phase=='cuda' else 'CPU_GPU_report.json')
                passed=code==0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
                exclusive(base/(phase+'.guard_receipt.json'),dict(passed=passed,exit_code=code,**usage,
                    exact_child_handle=handle,report_path=str(path),report_sha256=sha(path) if path.exists() else None))
                update(dict(active_pid=None))
                if not passed:raise RuntimeError('Preserved complete original CUDA preflight failed: '+phase)
        update(dict(state='original_fullbatch_CUDA_backward_and_complete_CPU_comparison_passed_training_still_pending'))
    except Exception as error:
        update(dict(state='failed_preserved_requires_isolated_repair',error=str(error)));raise
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--phase',choices=['cuda','cpu_reference']);a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.phase:(cuda_worker if a.phase=='cuda' else cpu_reference)(c)
    else:controller(c)


if __name__=='__main__':main()
