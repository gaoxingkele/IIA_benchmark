"""Full original-author CFM/OT-CFM/SB-CFM CUDA protocol, separate from CPU.

All120 source conditions, all24 real preflights, unchanged original package
lock/stopping rules, full native checkpoint replay and independent full W1.
"""
import argparse
import ast
import copy
from datetime import datetime,timezone
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as original
from scripts.flow_matching import audit_otcfm_singlecell_full_replay_v2 as replay
from scripts.flow_matching import otcfm_singlecell_result_integrity_v2 as integrity
from scripts.flow_matching import preflight_otcfm_singlecell_native_v4 as preflight
from scripts.flow_matching import preflight_crossad_original_cuda_full_batch_v1 as GPU_api

MODULE='scripts.flow_matching.run_otcfm_singlecell_original_CUDA_full120_v1'
CONFIG='configs/experiments/fm_otcfm_singlecell_original_CUDA_full120.v1.json'
FIRST='otcfm_singlecell__released_yaml__otcfm__released_author_donor32606__leave1__seed43'


def registration():
    base=read(ROOT/original.QUEUE);jobs=copy.deepcopy(base['jobs'])
    for job in jobs:
        canonical=job['id'];job.update(id=canonical+'__original_CUDA_FP32_v1',canonical_CPU_condition_reference=canonical,
            device_protocol='original_author_CUDA_FP32',model_config='configs/models/fm_otcfm_singlecell_original_CUDA__'+job['method']+'.v1.json',
            artifact_directory='F:/aicoding/IIA_Data/experiments/otcfm_singlecell_original_CUDA_full120_v1/'+hashlib.sha256(canonical.encode()).hexdigest()[:20])
    jobs.sort(key=lambda job:job['canonical_CPU_condition_reference']!=FIRST)
    c=copy.deepcopy(base)
    c.update(jobs=jobs,state_root='experiments/runs/fm_otcfm_singlecell_original_CUDA_full120_v1',
        resource_lock='experiments/runs/fm_original_cuda0_exclusive.lock',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_singlecell_original_CUDA_full120_v1',
        original_CPU_queue=original.QUEUE,original_CPU_queue_sha256=sha(ROOT/original.QUEUE),
        original_CPU_conditions_replaced=False,new_independent_training_seed_slots=0,
        required_real_preflight_conditions=24,TF32=False,autocast=False,execution_device='cuda:0',
        GPU=dict(device=0,minimum_free_bytes=18*2**30,emergency_free_bytes=2*2**30,allocator_fraction=.75,maximum_allocated_bytes=18*2**30),
        native_metric_tolerance=dict(rtol=2e-6,atol=2e-6),complete_CPU_GPU_field_tolerance=dict(rtol=2e-6,atol=2e-6))
    c['boundary']+=' Additional explicitly named original-author CUDA FP32 execution;all CPU cases and locks remain unchanged. No additional independent seeds,CPU completion credit or historical equivalence. Full24 actual batches/ODE/source paths and complete CPU/GPU initial velocity fields must pass before any full training. Every source fit must pass its actual guard and complete fresh native test/W1 LP audit before completion credit.'
    pins={r['path']:r['sha256'] for r in base['source_receipts']}
    for path in [original.QUEUE,MODULE.replace('.','/')+'.py','tests/test_fm_otcfm_singlecell_original_CUDA_full120_v1.py',
                 replay.CONFIG,replay.MODULE.replace('.','/')+'.py','scripts/flow_matching/otcfm_singlecell_result_integrity_v2.py',
                 preflight.MODULE.replace('.','/')+'.py',GPU_api.MODULE.replace('.','/')+'.py',
                 *['configs/models/fm_otcfm_singlecell_original_CUDA__'+m+'.v1.json' for m in base['methods']]]:
        pins[path]=sha(ROOT/path)
    c['source_receipts']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
    return c


def verify(c):
    if c!=registration():raise ValueError('All120 original source recipes,all24 preflights and unchanged author environment/gates required')
    for r in c['source_receipts']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Frozen native CUDA dependency changed')
    base={j['id']:j for j in read(ROOT/original.QUEUE)['jobs']}
    if len(c['jobs'])!=120 or {j['canonical_CPU_condition_reference'] for j in c['jobs']}!=set(base):raise ValueError('No original method/data/recipe/leaveout/seed may be omitted')
    for j in c['jobs']:
        for key in ['dataset','protocol','method','leaveout_timepoint','seed','scientific_protocol','net','data','source_yaml']:
            if j[key]!=base[j['canonical_CPU_condition_reference']][key]:raise ValueError('Original numerical/scientific field differs')


def configure_GPU(torch,policy):
    if not torch.cuda.is_available():raise ValueError('Actual original-author CUDA backend required;never fall back to CPU')
    if torch.is_autocast_enabled():raise ValueError('Original full FP32 precision required')
    torch.cuda.set_device(policy['device']);torch.cuda.set_per_process_memory_fraction(policy['allocator_fraction'],policy['device'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()


def transform_native(source):
    tree=ast.parse(source);counts=dict(environment=0,configure=0,placement=0,batches=0,times=0,trainer=0,guard=0,field=0)
    class Device(ast.NodeTransformer):
        function=None
        def visit_FunctionDef(self,node):
            before=self.function;self.function=node.name;node=self.generic_visit(node);self.function=before;return node
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if self.function=='source_runtime' and ast.unparse(node.targets[0])=="os.environ['CUDA_VISIBLE_DEVICES']":
                if ast.unparse(node.value)!="'-1'":raise ValueError('Pinned original CPU visibility assignment changed')
                node.value=ast.Constant('0');counts['environment']+=1
            if self.function=='worker' and len(node.targets)==1:
                name=ast.unparse(node.targets[0])
                if name=='(torch, pl, dm, model)':
                    counts['placement']+=1
                    return [node,*ast.parse("model = model.to('cuda:0')").body]
                if name in ['batch','validation_batch']:
                    node.value=ast.Call(func=ast.Name(id='_GPU_batch',ctx=ast.Load()),args=[node.value],keywords=[]);counts['batches']+=1
                    if name=='batch':
                        counts['field']+=1
                        return [node,*ast.parse('_observe_full_initial_velocity_field(model, batch)').body]
            return node
        def visit_Import(self,node):
            if self.function=='source_runtime' and any(n.name=='torch' for n in node.names):
                counts['configure']+=1;return [node,*ast.parse("_configure_GPU(torch, queue['GPU'])").body]
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if self.function=='worker' and ast.unparse(node.func)=='torch.linspace':
                counts['times']+=1
                return ast.Call(func=ast.Attribute(value=node,attr='to',ctx=ast.Load()),args=[ast.Constant('cuda:0')],keywords=[])
            if self.function=='worker' and ast.unparse(node.func)=='pl.Trainer':
                for keyword in node.keywords:
                    if keyword.arg=='accelerator' and ast.unparse(keyword.value)=="'cpu'":
                        keyword.value=ast.Constant('gpu');counts['trainer']+=1
            return node
        def visit_If(self,node):
            node=self.generic_visit(node)
            if self.function=='worker' and ast.unparse(node.test)=='torch.cuda.is_initialized()':
                node.test=ast.UnaryOp(op=ast.Not(),operand=node.test);counts['guard']+=1
            return node
    tree=Device().visit(tree)
    if counts!=dict(environment=1,configure=1,placement=1,batches=2,times=3,trainer=1,guard=1,field=1):
        raise ValueError('Only exact source device placements and stricter CUDA observations may change: '+str(counts))
    return ast.fix_missing_locations(tree)


def GPU_batch(batch):return [value.to('cuda:0') for value in batch]


def modules(c):
    native=types.ModuleType('_native_singlecell_original_CUDA_v1');native.__file__=original.__file__
    exec(compile(transform_native(Path(original.__file__).read_text(encoding='utf-8')),original.__file__+'::CUDA_only','exec'),native.__dict__)
    native.QUEUE=CONFIG;native.MODULE=MODULE;native._configure_GPU=configure_GPU;native._GPU_batch=GPU_batch
    native._observe_full_initial_velocity_field=lambda model,batch:None
    inspector=types.ModuleType('_native_singlecell_CUDA_integrity_v1');inspector.__file__=integrity.__file__
    exec(compile(Path(integrity.__file__).read_text(encoding='utf-8'),integrity.__file__,'exec'),inspector.__dict__)
    inspector.native=native
    tree=ast.parse(Path(replay.__file__).read_text(encoding='utf-8'));changes=[]
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef) or node.name!='worker':continue
        class ReplayDevice(ast.NodeTransformer):
            def visit_Call(self,n):
                n=self.generic_visit(n)
                if ast.unparse(n.func)=='pl.Trainer':
                    for k in n.keywords:
                        if k.arg=='accelerator' and ast.unparse(k.value)=="'cpu'":k.value=ast.Constant('gpu');changes.append('trainer')
                if ast.unparse(n.func)=='dict':
                    for k in n.keywords:
                        if k.arg=='cuda_context_initialized' and ast.unparse(k.value)=='False':k.value=ast.Constant(True);changes.append('proof')
                return n
            def visit_If(self,n):
                n=self.generic_visit(n)
                if ast.unparse(n.test)=='torch.cuda.is_initialized()':n.test=ast.UnaryOp(op=ast.Not(),operand=n.test);changes.append('guard')
                return n
        ReplayDevice().visit(node)
    if sorted(changes)!=['guard','proof','trainer']:raise ValueError('Every original full replay and independent W1 statement must remain')
    audit=types.ModuleType('_native_singlecell_CUDA_replay_v1');audit.__file__=replay.__file__
    exec(compile(ast.fix_missing_locations(tree),replay.__file__+'::CUDA_only','exec'),audit.__dict__)
    audit.native=native;audit.inspect=inspector.inspect;audit.CONFIG=CONFIG
    return native,inspector,audit


def probe_one(c,job):
    import numpy as np
    native,_,_=modules(c);proof=[];models=[]
    def observe(model,batch):
        import torch
        x=batch[0];t=torch.linspace(0.,1.,len(x))
        models.append(model);training=model.net.training
        reference=copy.deepcopy(model.net).cpu().eval();model.net.eval()
        with torch.no_grad():
            cpu=reference(t,x.detach().cpu()).detach().numpy()
            actual=model.net(t.to('cuda:0'),x).detach().cpu().numpy()
        model.net.train(training)
        if (cpu.shape!=(job['scientific_protocol']['batch_size'],5) or cpu.shape!=actual.shape
                or not np.isfinite(actual).all() or not np.allclose(actual,cpu,**c['complete_CPU_GPU_field_tolerance'])):
            raise ValueError('Every original full-batch initial velocity coordinate must pass fixed CPU/GPU comparison')
        out=Path(c['artifact_root'])/'preflight_fields'/hashlib.sha256(job['id'].encode()).hexdigest()[:20]
        out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'full_fields.npz',x=x.detach().cpu().numpy(),times=t.numpy(),CPU=cpu,CUDA=actual)
        proof.append(dict(compared_velocity_entries=int(cpu.size),full_batch_shape=list(x.shape),
            full_CPU_GPU_initial_velocity_compared=True,field_tolerance=c['complete_CPU_GPU_field_tolerance'],
            maximum_absolute_difference=float(np.max(np.abs(actual-cpu))),arrays_sha256=sha(out/'full_fields.npz')))
    native._observe_full_initial_velocity_field=observe
    value=native.worker(c,job,probe=True)
    if len(proof)!=1:raise ValueError('One full native field comparison required')
    import torch
    if (not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()>c['GPU']['maximum_allocated_bytes']
            or len(models)!=1 or any(p.device.type!='cuda' or p.dtype!=torch.float32
                or (p.grad is not None and (p.grad.dtype!=torch.float32 or not torch.isfinite(p.grad).all()))
                for p in models[0].parameters())):
        raise ValueError('Native actual FP32 CUDA and original GPU cap required')
    return dict(value,**proof[0],actual_CUDA=True,optimizer_updates=0,TF32=False,autocast=False,
        torch_version=torch.__version__,CUDA_build=torch.version.cuda,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())


def worker(c,phase,identity=None):
    if phase=='preflight':
        records=[]
        for job in preflight.cases(c):records.append(probe_one(c,job));gc.collect()
        exclusive(ROOT/c['state_root']/'full24_actual_CUDA_preflight.json',dict(status='passed',records=records,
            required_full_native_cases=24,model_optimizer_updates=0,benchmark_performance_claim=False,
            configuration_sha256=sha(ROOT/CONFIG),all_original_experiments_complete=False));return
    native,_,audit=modules(c);job=next(j for j in c['jobs'] if j['id']==identity)
    if phase=='train':native.worker(c,job)
    else:
        config=dict(c,artifact_root=c['artifact_root']+'/native_replays')
        audit.worker(config,c,job)


def GPU_wait(child,c,api,update):
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    violations=[];observations=dict(api)
    def snapshot():
        memory=api['resource_snapshot']()
        try:
            gpu=GPU_api.gpu_snapshot();update(dict(GPU=gpu))
            if gpu['free_bytes']<c['GPU']['emergency_free_bytes']:
                violations.append(gpu);api['terminate_tree'](child)
        except (OSError,ValueError,subprocess.SubprocessError) as error:update(dict(GPU_observation_error=repr(error)))
        return memory
    observations['resource_snapshot']=snapshot
    code,usage=capped_wait(child,c,observations,update)
    usage['GPU_resource_guard_aborted']=bool(violations);usage['resource_guard_aborted']|=bool(violations)
    return code,usage


def controller(c):
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    api=resource_api();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    lock=api['exclusive_lock'](base/'controller.lock')
    if lock is None or (base/'status.json').exists():
        if lock:lock.close()
        raise FileExistsError('Preserve every original native CUDA attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        verified_original_CUDA_ids=[],required_models=120,original_CPU_completion_credit=0,all_original_experiments_complete=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    def stage(phase,job=None):
        out=base/'preflight' if job is None else base/'attempts'/Path(job['artifact_directory']).name
        out.mkdir(parents=True,exist_ok=True)
        command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--phase',phase]
        if job is not None:command+=['--job-id',job['id']]
        with (out/(phase+'.stdout.log')).open('xb') as stdout,(out/(phase+'.stderr.log')).open('xb') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,
                PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0',MPLBACKEND='Agg',WANDB_MODE='disabled',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            update(dict(state='running_original_CUDA_'+phase,active_pid=child.pid,exact_child_handle=handle,
                active_job=job['id'] if job else None))
            code,usage=GPU_wait(child,c,api,update)
        path=base/'full24_actual_CUDA_preflight.json' if job is None else (Path(job['artifact_directory'])/'result.json' if phase=='train'
            else Path(c['artifact_root'])/'native_replays'/Path(job['artifact_directory']).name/'full_checkpoint_replay.json')
        passed=code==0 and not usage['resource_guard_aborted'] and path.exists()
        if passed:passed=read(path)['status']==('completed' if phase=='train' else 'passed')
        receipt=out/('guard_receipt.json' if phase=='train' else phase+'.guard_receipt.json')
        exclusive(receipt,dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
            result_sha256=sha(path) if path.exists() else None,report_path=str(path)))
        update(dict(active_pid=None))
        if not passed:raise RuntimeError('Preserved original native CUDA stage requires isolated repair: '+phase)
    try:
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
            if GPU_api.gpu_snapshot()['free_bytes']<c['GPU']['minimum_free_bytes']:raise RuntimeError('Original VRAM admission not met')
            stage('preflight')
        for job in c['jobs']:
            verify(c)
            with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
                if GPU_api.gpu_snapshot()['free_bytes']<c['GPU']['minimum_free_bytes']:raise RuntimeError('Original VRAM admission not met')
                stage('train',job);stage('replay',job)
            state['verified_original_CUDA_ids'].append(job['id']);update(dict(state='between_full_original_conditions'))
        update(dict(state='all120_original_CUDA_full_training_and_native_replays_complete_other_experiments_pending'))
    except Exception as error:update(dict(state='failed_preserved_requires_isolated_repair',error=repr(error)));raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--phase',choices=['preflight','train','replay']);p.add_argument('--job-id');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.phase:worker(c,a.phase,a.job_id)
    else:controller(c)


if __name__=='__main__':main()
