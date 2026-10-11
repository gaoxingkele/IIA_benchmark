"""Preserve all prior failed Energy artifacts and the full original70-job registry."""
import copy
from datetime import datetime,timezone
import json

import psutil

from scripts.flow_matching.run_spectral_energy42_preserved_full_retry_v1 import ROOT,read,sha,verify,IDENTITY
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    handle=psutil.Process(27184)
    if handle.create_time()!=1791697652.7172322 or not any('audit_spectral_full_trainer_sampling_graph_v1' in x for x in handle.cmdline()):
        raise ValueError('Actual full shared sampling verification owner must remain live')
    prior_path='configs/runtime/fm_spectral_original_resource_order.v4.json';prior=read(ROOT/prior_path)
    original=read(ROOT/prior['queue']);queue=copy.deepcopy(original)
    state='experiments/runs/fm_spectral_energy42_preserved_full_retry_v1';queue['state_root']=state
    job=next(j for j in queue['jobs'] if j['id']==IDENTITY)
    job['output_directory']=state+'/jobs/'+IDENTITY
    job['artifact_directory']='F:/aicoding/IIA_Data/experiments/spectral_energy42_preserved_full_retry_v1/'+IDENTITY
    queue_path='configs/experiments/fm_spectral_energy42_preserved_full_retry_queue.v1.json'
    exclusive_json(ROOT/queue_path,queue)
    preflight=read(ROOT/prior['preflight']);preflight['queue_sha256']=sha(ROOT/queue_path)
    preflight_path='configs/reproducibility/fm_spectral_energy42_preserved_full_retry_preflight.v1.json'
    exclusive_json(ROOT/preflight_path,preflight)
    shared=read(ROOT/'configs/reproducibility/fm_spectral_full_trainer_sampling_graph.2026-10-11_v1.json')
    config=dict(original_resource_order_configuration=prior_path,queue=queue_path,preflight=preflight_path,
        state_root=state,selected_output_directory=job['output_directory'],selected_artifact_directory=job['artifact_directory'],
        terminal_checkpoint_inspection=prior['measured_terminal_checkpoint_inspection'],poll_seconds=10,
        shared_sampling_receipt=shared['state_root']+'/receipt.json',
        shared_sampling_proof=shared['output_directory']+'/full_native_Trainer_sampling_graph_audit.json',
        expected_shared_sampling_controller=dict(pid=handle.pid,create_time=handle.create_time(),command=handle.cmdline()),
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273))
    sources={prior_path,prior['queue'],prior['preflight'],queue_path,preflight_path,config['terminal_checkpoint_inspection'],
        'scripts/flow_matching/run_spectral_energy42_preserved_full_retry_v1.py',
        'scripts/flow_matching/register_spectral_energy42_preserved_full_retry_v1.py',
        'tests/test_fm_spectral_energy42_preserved_full_retry_v1.py',
        'configs/reproducibility/fm_spectral_full_trainer_sampling_graph.2026-10-11_v1.json'}
    sources.update(pin['path'] for pin in prior['source_receipts'])
    sources.update(pin['path'] for pin in original['source_receipts'])
    inspection=read(ROOT/config['terminal_checkpoint_inspection'])
    sources.add(inspection['full_training_checkpoint_path'])
    config['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)]
    verify(config)
    path='configs/runtime/fm_spectral_energy42_preserved_full_retry.v1.json';exclusive_json(ROOT/path,config)
    evidence=ROOT/'projects/flow_matching_research/results/2026-10-12/spectral_energy42_preserved_full_retry_v1'
    exclusive_json(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=sha(ROOT/path),original_scope=config['original_scope'],
        full_original_jobs_retained=70,full_original_cohorts_retained=14,selected_failed_canonical_id=IDENTITY,
        previous25000step_checkpoint_preserved=True,previous_RNG_and_scheduler_state_saved=False,
        exact_warm_sampling_resume_certified=False,new_full_original_training_updates_required=25000,
        original_full_generated_shape=[20010,24,28],required_original_four_evaluator_repeats_each=5,
        existing_original_controller_unchanged=True,original_worker_sampler_and_guards_unchanged=True,
        shared_Stocks_CPU_graph_success_required_but_not_Energy_or_CUDA_capacity_certificate=True,
        full_actual_retry_executed=False,new_numeric_results_published=0,additional_training_seed_slots=0,
        all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,full_original_jobs_retained=70,full_original_training_updates_required=25000)))


if __name__=='__main__':main()
