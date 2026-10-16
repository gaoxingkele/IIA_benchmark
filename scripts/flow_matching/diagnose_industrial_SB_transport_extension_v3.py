"""Replay all twenty real training coupling problems, without any neural update."""
from pathlib import Path
import copy
import hashlib
import json
import psutil
import torch
import numpy as np
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.run_industrial_tsad_experiment import pooled_training_windows
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
from iia_benchmark.models.fm_objective_stable_extended import extended_plan

queue_path='configs/experiments/fm_industrial_empty_output_recovery_queue.v1.json'
q=read(ROOT/queue_path);case=q['jobs'][0];job=case['original_job']
assert case['original_id']=='fm_objective_sb__stable_numerics_v2__industrial_v1__pronto_day4__seed1103'
prior=ROOT/case['job']['output_directory'];failure=read(prior/'failure.json')
stderr=ROOT/q['settings']['output_root']/'logs'/case['job']['id']/'stderr.log'
assert failure['exit_code']==1 and not failure['resources']['resource_guard_aborted']
assert stderr.read_text(encoding='utf-8').rstrip().endswith('RuntimeError: Epsilon-continuation Sinkhorn did not converge: marginal error=1.03e-06')
assert not (prior/'model.pt').exists() and not (prior/'result.json').exists()
for pin in job['frozen_files']:assert sha(ROOT/pin['path'])==pin['sha256']
manifest=read(ROOT/job['dataset_manifest_path']);d=next(x for x in manifest['datasets'] if x['id']==job['dataset'])
with np.load(ROOT/d['input_npz'],allow_pickle=False) as inputs:
    data={k:inputs[k] for k in inputs.files}
training,groups=pooled_training_windows(data,d,100)
assert training.shape==(8,100,17) and sum(x['points'] for x in groups)==722
snapshot=resource_snapshot()
if snapshot['commit_available_bytes']<20*2**30:raise RuntimeError('Insufficient margin for bounded coupling-only diagnostic; full training retains32GiB')
torch.set_num_threads(2);generator=torch.Generator(device='cpu').manual_seed(1103)
tensor=torch.from_numpy(np.ascontiguousarray(training.reshape(8,-1)))
records=[];events=[];min_commit=snapshot['commit_available_bytes'];peak=0
for epoch in range(20):
    current=resource_snapshot()['commit_available_bytes'];min_commit=min(min_commit,current)
    if current<16*2**30:raise RuntimeError('Coupling-only diagnostic stopped at original emergency reserve')
    order=torch.randperm(8,generator=generator,device='cpu').cpu()
    x1=tensor[order].to('cpu');x0=torch.randn(x1.shape,generator=generator,device='cpu')
    plan=extended_plan(x0,x1,2.,audit=events)
    weights=torch.as_tensor(plan.ravel(),dtype=torch.float64,device='cpu')
    pairs=torch.multinomial(weights,8,replacement=True,generator=generator)
    # The original network is Linear/SiLU only and does not consume this local
    # generator. Consume the exact remaining source RNG calls without inference.
    torch.rand(8,generator=generator,device='cpu').clamp(1e-4,1-1e-4)
    torch.randn(x1.shape,generator=generator,device='cpu')
    records.append(dict(epoch=epoch+1,x0_sha256=hashlib.sha256(x0.numpy().tobytes()).hexdigest(),
        x1_sha256=hashlib.sha256(x1.numpy().tobytes()).hexdigest(),coupling_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
        pair_indices=pairs.tolist(),**events[-1]))
    peak=max(peak,psutil.Process().memory_info().private)
    if peak>2*2**30:raise RuntimeError('Bounded coupling diagnostic exceeded2GiB private cap')
    print(json.dumps(dict(epoch=epoch+1,used_extended_cap=events[-1]['used_extended_cap'],marginal_error=events[-1]['actual_absolute_marginal_error'])),flush=True)
assert len(records)==20 and any(x['used_extended_cap'] for x in records)
first=next(x for x in records if x['used_extended_cap'])
assert first['original_failure'].endswith('marginal error=1.03e-06')
assert all(x['tolerance']==1e-6 and x['regularization']==2. and x['actual_absolute_marginal_error']<=1e-6 for x in records)
out=Path(r'F:\aicoding\IIA_Data\experiments\industrial_SB_transport_extension_v3');out.mkdir(parents=True,exist_ok=False)
write(out/'all20_real_coupling_problems.json',dict(passed=True,original_case=job,original_queue=queue_path,original_queue_sha256=sha(ROOT/queue_path),
    original_failure_sha256=sha(prior/'failure.json'),original_stderr_sha256=sha(stderr),input_sha256=sha(ROOT/d['input_npz']),
    fit_shape=list(training.shape),unique_fit_groups=groups,coupling_problems=records,actual_first_original_nonconvergence_epoch=first['epoch'],
    final_epsilon=2.,unchanged_absolute_marginal_tolerance=1e-6,original_final_iteration_cap=20000,new_final_iteration_cap=200000,
    peak_private_bytes=peak,minimum_commit_available_bytes=min_commit,diagnostic_initial_commit_requirement=20*2**30,
    original_full_training_admission_commit_requirement=32*2**30,diagnostic_is_not_full_training=True,
    new_neural_forwards=0,new_optimizer_updates=0,new_training_seeds=0,full_benchmark_performance_complete=False,
    source_hashes={p:sha(ROOT/p) for p in ['src/iia_benchmark/models/fm_objective_stable.py','src/iia_benchmark/models/fm_objective_stable_extended.py','tests/test_fm_objective_stable_extended.py']},
    all_original_experiments_complete=False))
for name in ['sb','sf2m','sf2m_flow_only']:
    old_path=f'configs/models/fm_objective_{name}_stable_v2.json';old=read(ROOT/old_path)
    path=ROOT/f'configs/models/fm_objective_{name}_stable_extended_v3.json';new=read(path)
    parameters=dict(new['parameters']);assert parameters.pop('sinkhorn_max_final_iterations')==200000
    assert parameters==old['parameters'] and new['numerical_repair']['original_config_sha256']==sha(ROOT/old_path)
    assert new['entrypoint']=='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector'
print(json.dumps(dict(output=str(out),first_original_failure_epoch=first['epoch'],all20_couplings_verified=True,new_full_training_results=0)),flush=True)
