"""Full original GECCO seed1105 CUDA training, source evaluation and fresh replay.

Separate explicit device protocol. Existing CPU cases remain immutable and
pending under their own IDs. No reduced epochs, batches, precision or metrics.
"""
import argparse
import ast
import copy
from datetime import datetime,timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import preflight_crossad_original_cuda_full_batch_v1 as preflight

MODULE='scripts.flow_matching.run_crossad_original_cuda_first_full_v1'
CONFIG='configs/experiments/fm_crossad_original_cuda_first_full.v1.json'
MODEL='configs/models/fm_crossad_original_cuda_first_full.v1.json'


def registration():
    base=read(ROOT/preflight.CONFIG);q=read(ROOT/preflight.QUEUE)
    job=copy.deepcopy(base['job']);canonical=job['id']
    job.update(id=canonical+'__original_CUDA_FP32_v1',canonical_CPU_condition_reference=canonical,
        device_protocol='original_CUDA_FP32',activation_checkpointing=False,
        runner=MODULE,output_directory='experiments/runs/fm_crossad_original_cuda_first_full_v1/author_job')
    settings=copy.deepcopy(q['settings']);settings.update(runtime_root=base['original_source_root'],
        output_root='experiments/runs/fm_crossad_original_cuda_first_full_v1')
    settings['boundary'] += ' Additional original-source CUDA FP32 protocol;canonical CPU conditions remain separate and untouched. Original GPU masks and full model are used without activation replay. Original20epoch maximum/earlystop,batch128,full splits,validation selection and complete source evaluation retained. TF32/autocast are disabled. New explicit execution ID is not an independent additional training seed,CPU completion credit,strictTAB result or historical framework-equivalence proof.'
    paths=[MODULE.replace('.','/')+'.py',MODEL,preflight.CONFIG,preflight.MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/run_crossad_author_job.py','tests/test_fm_crossad_original_cuda_first_full_v1.py']
    paths += [r['path'] for r in base['source_receipts']]
    for name in ['CUDA_report.json','CPU_GPU_report.json']:
        paths.append(str(Path(base['artifact_root'])/name))
    for phase in ['cuda','cpu_reference']:paths.append(base['state_root']+'/'+phase+'.guard_receipt.json')
    return dict(job=job,settings=settings,original_queue=preflight.QUEUE,canonical_CPU_condition_reference=canonical,
        actual_original_epochs_max=20,original_early_stop_patience=3,original_batch_size=128,parameters=1327625,
        source_native_batch_preflight=preflight.CONFIG,GPU=base['GPU'],host_guards=base['settings'],
        python=base['python'],cpu_threads=base['cpu_threads'],resource_lock=base['resource_lock'],
        array_tolerance=base['CPU_GPU_array_tolerance'],new_independent_training_seed_slots=0,
        state_root='experiments/runs/fm_crossad_original_cuda_first_full_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_original_cuda_first_full_v1',
        original_scope=base['original_scope'],required_conditions=1,other_original_conditions_removed=False,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Full original model,budget,source,device protocol and numerical proofs required')
    base=read(ROOT/preflight.CONFIG)
    for phase,name in [('cuda','CUDA_report.json'),('cpu_reference','CPU_GPU_report.json')]:
        path=Path(base['artifact_root'])/name;guard=read(ROOT/base['state_root']/(phase+'.guard_receipt.json'))
        if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
                or guard['report_sha256']!=sha(path) or not read(path)['passed']):
            raise ValueError('Actual original fullbatch CUDA backward and complete CPU comparison must pass')


def configured_functions(original):
    tree=ast.parse(inspect.getsource(original.run));changed=[]
    class Device(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='SimpleNamespace':
                for kw in node.keywords:
                    if kw.arg=='use_gpu' and isinstance(kw.value,ast.Constant) and kw.value.value is False:
                        kw.value=ast.Constant(True);changed.append('use_gpu')
                    elif kw.arg=='gpu_type' and isinstance(kw.value,ast.Constant) and kw.value.value=='cpu':
                        kw.value=ast.Constant('cuda');changed.append('gpu_type')
                    elif kw.arg=='device' and ast.unparse(kw.value)=="torch.device('cpu')":
                        kw.value=ast.parse("torch.device('cuda:0')",mode='eval').body;changed.append('device')
            return node
    tree=Device().visit(tree)
    if sorted(changed)!=['device','gpu_type','use_gpu']:raise ValueError('Exactly three original execution-device fields may change')
    scores=ast.parse(inspect.getsource(original.native_scores));count=0
    class Scores(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if ast.unparse(node.func)=='torch.as_tensor':
                if any(k.arg=='device' for k in node.keywords):raise ValueError('Original score device contract changed')
                node.keywords.append(ast.keyword(arg='device',value=ast.parse('next(model.parameters()).device',mode='eval').body));count+=1
            return node
    scores=Scores().visit(scores)
    if count!=2:raise ValueError('Both original scoring tensor allocations must follow model device')
    namespace=dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(scores),__file__+'::native_score_device','exec'),namespace)
    exec(compile(ast.fix_missing_locations(tree),__file__+'::original_CUDA_run','exec'),namespace)
    return namespace['run'],namespace['native_scores']


def precision(c):
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise ValueError('Original CUDA run cannot fall back to CPU')
    torch.set_num_threads(c['cpu_threads'])
    torch.backends.fp32_precision='ieee';torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.cuda.set_per_process_memory_fraction(c['GPU']['allocator_fraction'],0)
    random.seed(c['job']['seed']);np.random.seed(c['job']['seed']);torch.manual_seed(c['job']['seed']);torch.cuda.manual_seed_all(c['job']['seed'])
    torch.cuda.reset_peak_memory_stats()
    return torch,np


def compare_nested(a,b,tolerance):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('Complete metric fields must match')
        for k in a:compare_nested(a[k],b[k],tolerance)
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):raise ValueError('Every metric list entry required')
        for x,y in zip(a,b):compare_nested(x,y,tolerance)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=tolerance['rtol'],abs_tol=tolerance['atol']):
            raise ValueError('A complete metric or uncertainty value differs')
    elif a!=b:raise ValueError('Metric protocol/provenance/coverage fields differ')


def worker(c,phase):
    torch,np=precision(c)
    from scripts.flow_matching import run_crossad_author_job as original
    run,scores=configured_functions(original)
    job,settings=c['job'],c['settings'];out=ROOT/job['output_directory']
    if not out.is_junction() or out.resolve()!=Path(c['artifact_root']).resolve()/'author_job':
        raise ValueError('Large new author artifacts must follow the registered fresh F-drive junction')
    if phase=='train':
        run(job,settings)
        source=read(out/'result.json')
        if source['details']['declared_train_parameters']!=job['train_parameters']:
            raise ValueError('Original training recipe changed')
        exclusive(ROOT/c['state_root']/'source_execution_device_proof.json',dict(passed=True,
            source_result_sha256=sha(out/'result.json'),original_CUDA_source=True,activation_checkpointing=False,
            torch_version=torch.__version__,CUDA_build=torch.version.cuda,device=torch.cuda.get_device_name(0),
            FP32_precision=torch.backends.cuda.matmul.fp32_precision,autocast_enabled=torch.is_autocast_enabled('cuda'),
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
            canonical_CPU_condition_reference=c['canonical_CPU_condition_reference'],new_independent_training_seed_slots=0,
            all_original_experiments_complete=False))
        return
    source=read(out/'result.json');details=source['details']
    if source['experiment_sha256']!=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest():
        raise ValueError('Full original GPU source result identity changed')
    for item in source['artifacts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen source result artifact changed')
    from iia_benchmark.models.fm_crossad_author_runtime import build_model,install_wide_cache,load_release
    runtime=ROOT/settings['runtime_root'];install_wide_cache(runtime,ROOT/settings['cache_root'])
    provider=__import__('importlib').import_module('data_provider.data_provider')
    data,_=provider.data_provider(str(ROOT/settings['raw_root']),job['dataset'],128,128,1,'train')
    model=build_model(runtime,job['model_parameters']).float().to('cuda:0')
    checkpoint=out/'workspace/configs'/job['dataset']/'checkpoints_0/checkpoint.pth'
    loaded=load_release(model,checkpoint);model.eval()
    target=Path(c['artifact_root'])/'native_replay';target.mkdir(parents=True,exist_ok=False)
    saved=np.load(out/'native_scores.npz',allow_pickle=False);actual={};proof=[]
    for name,values in [('train',data.train),('validation',data.val),('test',data.test)]:
        result=scores(model,values,job['model_parameters']['seq_len'],job['train_parameters']['batch_size'])
        expected=saved[name]
        if result.dtype!=expected.dtype or result.shape!=expected.shape or result.tobytes()!=expected.tobytes():
            raise ValueError('Every full original CUDA score must replay byte-for-byte from the saved model')
        np.save(target/(name+'_scores.npy'),result,allow_pickle=False)
        proof.append(dict(role=name,points=len(result),dtype=str(result.dtype),all_score_bytes_equal=True,
            score_file_sha256=sha(target/(name+'_scores.npy'))));actual[name]=result
    labels=data.test_label.reshape(-1).astype(np.uint8)
    if labels.tobytes()!=saved['labels'].tobytes() or not np.array_equal(saved['source_rows'],np.arange(len(labels))+job['train_length']):
        raise ValueError('Every original test label and source-row identity required')
    if (details['parameter_count']!=c['parameters'] or not 1<=details['epochs_completed']<=20
            or details['training_forward_calls']!=details['epochs_completed']*432):
        raise ValueError('All original full training epochs/batches and source earlystop budget required')
    repeated=original.evaluate_saved({job['dataset']:(labels,actual['test'])},actual['validation'],actual['train'],
        job['raw_sha256'],sha(checkpoint),settings['evaluation'],job['seed'])
    compare_nested(source['metrics']['native_validation_control'],repeated,c['array_tolerance'])
    exclusive(target/'report.json',dict(passed=True,source_result_sha256=sha(out/'result.json'),
        job=job,checkpoint=loaded,all_roles=proof,full_test_labels_and_source_rows_equal=True,
        actual_epochs=details['epochs_completed'],actual_training_forward_calls=details['training_forward_calls'],
        full_native_validation_metrics_and_uncertainty_recomputed=True,native_validation_control=repeated,
        strict_TAB_result=False,paper_equivalence=False,canonical_CPU_condition_marked_complete=False,
        new_independent_training_seed_slots=0,all_original_experiments_complete=False))


def controller(c):
    import psutil
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True);owner=exclusive_lock(base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Preserve every original full CUDA attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,completed_conditions=0,
        canonical_CPU_condition_reference=c['canonical_CPU_condition_reference'],all_original_experiments_complete=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while preflight.gpu_snapshot()['free_bytes']<c['GPU']['minimum_free_bytes']:
            update(dict(state='waiting_registered_free_VRAM',GPU=preflight.gpu_snapshot()));time.sleep(10)
        with resource_slot(ROOT/c['resource_lock'],c['host_guards'],update):
            if preflight.gpu_snapshot()['free_bytes']<c['GPU']['minimum_free_bytes']:raise RuntimeError('VRAM changed before ownership;no worker launched')
            for phase in ['train','replay']:
                verify(c)
                command=[str(ROOT/c['python']),'-B','-X','utf8','-u','-m',MODULE,'--phase',phase]
                with (base/(phase+'.stdout.log')).open('xb') as out,(base/(phase+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        OMP_NUM_THREADS=str(c['cpu_threads']),MKL_NUM_THREADS=str(c['cpu_threads'])))
                    handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                    update(dict(state='running_original_CUDA_full_'+phase,active_pid=child.pid,exact_child_handle=handle))
                    code,usage=preflight.guarded_GPU_wait(child,dict(GPU=c['GPU'],settings=c['host_guards']),update)
                path=ROOT/c['job']['output_directory']/'result.json' if phase=='train' else Path(c['artifact_root'])/'native_replay/report.json'
                passed=code==0 and not usage['resource_guard_aborted'] and path.exists()
                if phase=='replay' and passed:passed=read(path)['passed']
                exclusive(base/(phase+'.guard_receipt.json'),dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
                    report_path=str(path),report_sha256=sha(path) if path.exists() else None))
                update(dict(active_pid=None))
                if not passed:raise RuntimeError('Preserved full original CUDA attempt failed: '+phase)
        update(dict(state='full_original_CUDA_condition_and_checkpoint_replay_complete_other_original_conditions_still_pending',completed_conditions=1))
    except Exception as error:
        update(dict(state='failed_preserved_requires_isolated_repair',error=repr(error)));raise
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--phase',choices=['train','replay']);a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.phase:worker(c,a.phase)
    else:controller(c)


if __name__=='__main__':main()
