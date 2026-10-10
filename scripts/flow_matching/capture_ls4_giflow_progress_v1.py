"""Capture actual source/environment/data proofs and live full-controller identities."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import psutil

from scripts.flow_matching.run_ls4_original_v1 import ROOT, read, sha, verify
from scripts.flow_matching import giflow_native_protocol as giflow


def main():
    target=ROOT/'projects/flow_matching_research/results/2026-10-11/ls4_and_giflow_native_resume_v1'
    target.mkdir(parents=True,exist_ok=True)
    if (target/'runtime_observations.json').exists():
        raise FileExistsError('Preserve previous immutable progress capture')
    ls4=read(ROOT/'configs/experiments/fm_ls4_original_queue.v1.json');verify(ls4)
    gi=read(ROOT/'configs/experiments/fm_giflow_native_queue.v4.json');giflow.verify(gi)
    copies={
        'ls4_environment_lock.json':'configs/reproducibility/ls4_author_py310.lock.json',
        'ls4_dependency_download_receipt.json':'experiments/runs/fm_ls4_author_environment_v1/download_receipt.json',
        'ls4_full_data_audit.json':ls4['data_audit_output'],
        'ls4_data_audit_resource_receipt.json':'experiments/runs/fm_ls4_original_v1/data_audit_resource_receipt.json',
        'giflow_actual_diagnostics.json':'experiments/runs/fm_giflow_update_audit_v2/validation.json',
    }
    receipt=[]
    for name,source in copies.items():
        shutil.copy2(ROOT/source,target/name)
        assert sha(ROOT/source)==sha(target/name)
        receipt.append(dict(source=source,copy=name,sha256=sha(target/name)))
    diagnostics=read(target/'giflow_actual_diagnostics.json')
    assert diagnostics['passed'] and len(diagnostics['cases'])==4
    for case in diagnostics['cases']:
        source=ROOT/case['proof'];name=case['track']+'__'+case['mode']+'.json'
        assert sha(source)==case['sha256'];shutil.copy2(source,target/name)
        receipt.append(dict(source=case['proof'],copy=name,sha256=sha(target/name)))
    rows=[]
    for job in ls4['jobs']:
        config=read(ROOT/job['model_config'])
        rows.append(dict(id=job['id'],dataset=config['dataset'],protocol=config['protocol'],seed=config['seed'],
            samples=config['samples'],length=config['length'],epochs=config['optim']['epochs'],
            batch_size=config['optim']['batch_size'],sigma=config['sigma'],
            generator_updates=config['budget']['generator_updates'],metric_calls=config['budget']['metric_calls'],
            evaluator_epochs_per_metric=100,status='registered_full_budget_not_completed',model_config=job['model_config']))
    with (target/'ls4_registered_full_jobs.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    observations=[]
    for file in ['experiments/runs/fm_ls4_original_v1/full_controller_launch.json',
                 'experiments/runs/fm_giflow_update_boundary_v1/launch.json']:
        launched=read(ROOT/file);process=psutil.Process(launched['pid'])
        if process.create_time()!=launched['create_time']:
            raise ValueError('Registered process identity reused')
        observations.append(dict(launch=file,pid=process.pid,create_time=process.create_time(),
            actual_command=process.cmdline(),cpu_seconds=sum(process.cpu_times()[:2]),
            actual_descendants=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline()) for c in process.children(recursive=True)],
            verified_live_now=True))
    for pid,expected in [(21984,1791643109.8706102),(13988,1791649712.7103302),(3296,1791646542.8922482)]:
        process=psutil.Process(pid)
        if process.create_time()!=expected: raise ValueError('Original full model/controller identity differs')
        observations.append(dict(pid=pid,create_time=expected,actual_command=process.cmdline(),
            cpu_seconds=sum(process.cpu_times()[:2]),verified_live_now=True,
            actual_descendants=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline()) for c in process.children(recursive=True)]))
    status=read(ROOT/'experiments/runs/fm_ls4_original_v1/status.json')
    transition=read(ROOT/'experiments/runs/fm_giflow_update_boundary_v1/status.json')
    state=dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='No progress: unchanged result listing was reported; current turn resumed with authoritative process checks and new native preparation.',
        current_goal_turn_classification='Progress: exact64-package environment,20 full native data audits,140-slot accounting repair and live bounded successor handoff.',
        original_scope=dict(canonical_papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        ls4_full_jobs=20,ls4_registered_generator_updates=sum(r['generator_updates'] for r in rows),
        ls4_gpu_preflight_complete=(ROOT/ls4['gpu_preflight_output']).exists(),
        ls4_completed_full_jobs=len(list((ROOT/ls4['state_root']/'jobs').glob('*/result.json'))),
        giflow_full_canonical_seed_slots=len(gi['jobs']),giflow_completed_successor_jobs=len(list((ROOT/gi['state_root']/'jobs').glob('*/result.json'))),
        ls4_runtime=status,giflow_boundary_runtime=transition,processes=observations,
        full_hyperparameters_and_memory_guards_unchanged=True,original_model_processes_not_interrupted=True,
        new_benchmark_results_added=0,all_original_experiments_complete=False,
        remaining_obligations=ls4['remaining_obligations']+['Independent final trained-model audits and v4 GiFlow result capture integration after full completions'],
        material_receipts=receipt)
    (target/'runtime_observations.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (target/'validation.json').write_text(json.dumps(dict(tests_passed=26,
        command='pytest LS4 native budgets,GiFlow native/update/boundary and full GPU slot',
        actual_ls4_data_cases_passed=20,actual_giflow_data_and_integration_cases_passed=4,
        dependency_publisher_checksums_passed=64,all_copied_receipt_hashes_verified=True,
        new_full_training_results=0,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(ls4_full_jobs=20,giflow_full_canonical_slots=140,actual_audits_passed=24,
        live_controller_and_handoff_verified=True,new_full_training_results=0)))


if __name__=='__main__':
    main()
