"""Bounded coupling-only qualification for the eight remaining stable cases.

Local source RNG is independent of the Linear/SiLU network and Adam. Reuse
the already qualified day4/1103 sequence; execute two other full sequences.
No network, optimizer, original output or original training guard is changed.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.diagnose_industrial_remaining8_entropic_couplings_v1'
TEST='tests/test_fm_industrial_remaining8_entropic_couplings_v1.py'
CONFIG='configs/reproducibility/fm_industrial_remaining8_entropic_couplings.v1.json'
QUEUE='configs/experiments/fm_industrial_empty_output_recovery_queue.v1.json'
PRIOR='F:/aicoding/IIA_Data/experiments/industrial_SB_transport_extension_v3/all20_real_coupling_problems.json'
PRIOR_GUARD='experiments/runs/fm_industrial_SB_transport_extension_v3/attempts/coupling_diagnostic/receipt.json'


def group_cases(queue,manifest,models):
    groups={}
    for case in queue['jobs']:
        j=case['original_job'];p=models[j['model_config']]['parameters'];d=next(d for d in manifest['datasets'] if d['id']==j['dataset'])
        if (j['parameter_overrides']!=dict(seed=j['seed'],device='cpu') or j['window']!=p['window']
                or p['epochs']!=20 or p['batch_size']!=64 or p['sigma']!=1. or p['objective'] not in ['sb','sf2m']):
            raise ValueError('Original full CPU coupling budget and parameters required')
        count=sum((g['points']-j['window'])//j['window']+1+int((g['points']-j['window'])%j['window']!=0) for g in d['training_groups'])
        key=j['dataset']+'__seed'+str(j['seed'])
        item=groups.setdefault(key,dict(dataset=j['dataset'],seed=j['seed'],representative_job=j,covered_original_ids=[],
            fit_points=d['scaler']['fit_points'],fit_windows=count,expected_coupling_problems=20*math.ceil(count/64),
            reuse_existing=(key=='pronto_day4__seed1103')))
        # Only objective and auxiliary score loss can differ; neither consumes the local Generator.
        a=models[item['representative_job']['model_config']]['parameters']
        ignore={'objective','score_weight'}
        if {k:v for k,v in p.items() if k not in ignore}!={k:v for k,v in a.items() if k not in ignore}:
            raise ValueError('Shared sequence must have identical data/window/batch/epoch/RNG parameters')
        item['covered_original_ids'].append(case['original_id'])
    expected={'pronto_day4__seed1103':(8,20),'pronto_day4__seed1104':(8,20),'pronto_day3__seed1107':(70,40)}
    if set(groups)!=set(expected) or any((g['fit_windows'],g['expected_coupling_problems'])!=expected[k] or len(g['covered_original_ids'])!=3 for k,g in groups.items()):
        raise ValueError('All nine original cases, including eight remaining cases, and all original windows required')
    return groups


def registration():
    q=read(ROOT/QUEUE);manifest_path=q['jobs'][0]['original_job']['dataset_manifest_path'];manifest=read(ROOT/manifest_path)
    models={j['original_job']['model_config']:read(ROOT/j['original_job']['model_config']) for j in q['jobs']}
    settings=dict(read(ROOT/'configs/runtime/fm_industrial_SB_transport_extension.v3.json')['diagnostic_settings'])
    paths=[MODULE.replace('.','/')+'.py',TEST,QUEUE,manifest_path,PRIOR,PRIOR_GUARD,
        'configs/runtime/fm_industrial_SB_transport_extension.v3.json',
        'scripts/flow_matching/run_industrial_tsad_experiment.py',
        'src/iia_benchmark/models/fm_objective_stable.py','src/iia_benchmark/models/fm_objective_stable_extended.py',
        'src/iia_benchmark/models/fm_objective_detectors.py','scripts/flow_matching/cached_resource_snapshot_v1.py']+list(models)
    return dict(queue=QUEUE,groups=group_cases(q,manifest,models),prior_qualified_report=PRIOR,prior_guard=PRIOR_GUARD,
        remaining_original_cases=[x['original_id'] for x in q['jobs'][1:]],
        state_root='experiments/runs/fm_industrial_remaining8_entropic_couplings_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_remaining8_entropic_couplings_v1',
        resource_lock='experiments/runs/fm_bounded_transport_math_diagnostic_cpu.lock',settings=settings,
        post_import_commit_headroom_bytes=20*2**30,original_full_training_commit_admission_bytes=32*2**30,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        expected_new_numerical_sequences=2,expected_new_coupling_problems=60,prior_coupling_problems_reused=20,
        original_CPU_scope=550,original_GPU_scope=850,
        original_global_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        literature_citation='https://arxiv.org/abs/2302.00482',dataset_citation='configs/datasets/pronto_public.json',
        full_original_training_complete=False,new_neural_forwards=0,new_optimizer_updates=0,new_training_seed_slots=0,
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('All frozen source budgets, couplings, cases and original guards must remain')
    q=read(ROOT/QUEUE)
    for case in q['jobs']:
        for p in case['original_job']['frozen_files']:
            if sha(ROOT/p['path'])!=p['sha256']:raise ValueError('Frozen source changed: '+p['path'])
    r,g=read(c['prior_qualified_report']),read(ROOT/c['prior_guard'])
    if not r['passed'] or len(r['coupling_problems'])!=20 or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(c['prior_qualified_report']):
        raise ValueError('Only the already qualified complete real sequence can be reused')
    return q


def sequence(windows,parameters,seed,solve,observe=None):
    """Exact source randperm/x0/multinomial/time/noise order, no neural calls."""
    import numpy as np
    import torch
    tensor=torch.from_numpy(np.ascontiguousarray(windows.reshape(len(windows),-1)))
    generator=torch.Generator(device='cpu').manual_seed(seed);records=[]
    for epoch in range(parameters['epochs']):
        order=torch.randperm(len(windows),generator=generator,device='cpu').cpu()
        for start in range(0,len(windows),parameters['batch_size']):
            x1=tensor[order[start:start+parameters['batch_size']]].to('cpu')
            x0=torch.randn(x1.shape,generator=generator,device='cpu')
            record=dict(epoch=epoch+1,batch_start=start,batch_size=len(x1),regularization=2*parameters['sigma']**2,
                x0_sha256=hashlib.sha256(x0.numpy().tobytes()).hexdigest(),x1_sha256=hashlib.sha256(x1.numpy().tobytes()).hexdigest(),
                local_generator_before_transport_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            try:plan,audit=solve(x0,x1,2*parameters['sigma']**2)
            except Exception as error:
                if observe:observe(dict(record,error=str(error)),x0,x1)
                raise
            weights=torch.as_tensor(plan.ravel(),dtype=torch.float64,device='cpu')
            pairs=torch.multinomial(weights,len(x0),replacement=True,generator=generator)
            x0,x1=x0[pairs//len(x1)],x1[pairs%len(x1)]
            torch.rand(len(x1),generator=generator,device='cpu').clamp(1e-4,1-1e-4)
            torch.randn(x1.shape,generator=generator,device='cpu')
            record.update(audit,coupling_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),pair_indices=pairs.tolist(),
                local_generator_after_problem_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            records.append(record)
            if observe:observe(record,x0,x1)
    return records


def worker(c,key):
    verify(c)
    import numpy as np
    import torch
    import psutil
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.run_industrial_tsad_experiment import pooled_training_windows
    from iia_benchmark.models.fm_objective_stable_extended import extended_plan
    if resource_snapshot()['commit_available_bytes']<c['post_import_commit_headroom_bytes']:raise RuntimeError('Original post-import math reserve not met')
    torch.set_num_threads(2);group=c['groups'][key];job=group['representative_job']
    manifest=read(ROOT/job['dataset_manifest_path']);dataset=next(d for d in manifest['datasets'] if d['id']==job['dataset'])
    if sha(ROOT/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Full frozen native input changed')
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as arrays:data={k:arrays[k].copy() for k in arrays.files}
    windows,budget=pooled_training_windows(data,dataset,job['window'])
    if len(windows)!=group['fit_windows'] or sum(g['points'] for g in budget)!=group['fit_points']:raise ValueError('All original fit coordinates/windows required')
    target=Path(c['artifact_root'])/key;target.mkdir(parents=True,exist_ok=False)
    successful=[]
    def observe(record,x0,x1):
        if record.get('error'):
            np.savez_compressed(target/'failed_original_endpoints.npz',x0=x0.numpy(),x1=x1.numpy())
            write(target/'preserved_failure.json',dict(record=record,previous_completed_problems=successful,
                original_final_cap=20000,extended_final_cap=200000,original_tolerance=1e-6,no_retry_or_resampling=True))
        else:successful.append(record)
        write(target/'progress.json',dict(completed_problems=len(successful),latest=record,new_neural_forwards=0,new_optimizer_updates=0))
    def solve(x0,x1,epsilon):
        if resource_snapshot()['commit_available_bytes']<16*2**30:raise RuntimeError('Original emergency math reserve not met')
        if psutil.Process().memory_info().private>2*2**30:raise RuntimeError('Original two-GiB bounded math cap exceeded')
        audit=[];plan=extended_plan(x0,x1,epsilon,extended_iterations=200000,audit=audit)
        return plan,audit[-1]
    parameters=read(ROOT/job['model_config'])['parameters']
    records=sequence(windows,parameters,job['seed'],solve,observe)
    if len(records)!=group['expected_coupling_problems'] or any(r['actual_absolute_marginal_error']>1e-6 or r['regularization']!=2. for r in records):
        raise ValueError('Every original real coupling and unchanged target must pass')
    write(target/'report.json',dict(passed=True,group=group,fit_shape=list(windows.shape),fit_group_budget=budget,
        input_sha256=dataset['input_sha256'],coupling_problems=records,used_extended_cap_problems=sum(r['used_extended_cap'] for r in records),
        maximum_absolute_marginal_error=max(r['actual_absolute_marginal_error'] for r in records),
        new_neural_forwards=0,new_optimizer_updates=0,full_original_model_training_complete=False,all_original_experiments_complete=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');p.add_argument('--group');a=p.parse_args()
    if a.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:
        print(json.dumps(dict(passed=True,remaining_cases=8,new_sequences=2,new_problems=60,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    if a.group:worker(c,a.group);return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():raise RuntimeError('Preserve existing mathematical attempts and owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,qualified_groups=['pronto_day4__seed1103'],failed_preserved={},
        prior_report_sha256=sha(c['prior_qualified_report']),new_neural_forwards=0,new_optimizer_updates=0,all_original_experiments_complete=False)
    def update(v):
        v=dict(v)
        if v.get('state')=='replaying_full_checkpoints':v['state']='solving_full_original_coupling_sequence'
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        for key,group in c['groups'].items():
            if group['reuse_existing']:continue
            with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
                attempt=base/'attempts'/key;attempt.mkdir(parents=True,exist_ok=False)
                command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--group',key]
                with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    update(dict(state='solving_full_original_coupling_sequence',active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),active_group=key,actual_command=command))
                    code,usage=capped_wait(child,c,api,update)
                proof=Path(c['artifact_root'])/key/'report.json'
                passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                receipt=dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None)
                write(attempt/'receipt.json',receipt)
                if passed:state['qualified_groups'].append(key)
                else:state['failed_preserved'][key]=receipt
                update(dict(active_pid=None,state='group_attempt_terminal'))
        update(dict(state='all_group_attempts_terminal',active_pid=None))
    finally:owner.close()


if __name__=='__main__':main()
