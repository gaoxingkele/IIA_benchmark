"""Register every original RCGAN seed for complete500 checkpoint/generator/MMD replay."""
import copy
from datetime import datetime,timezone
import json
import psutil

from scripts.flow_matching.run_rcgan_full500_completion_replay_queue_v1 import ROOT,read,sha,verify,exclusive


def main():
    original_path='configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json'
    queue=read(ROOT/original_path)
    recovery_paths={1103:'configs/reproducibility/fm_rcgan_seed1103_preserved_full262_retry.2026-10-12_v1.json',
        1104:'configs/reproducibility/fm_rcgan_seed1104_full416_state_recovery.2026-10-11_v2.json'}
    source_paths={original_path,'scripts/flow_matching/run_rcgan_full500_completion_replay_queue_v1.py',
        'scripts/flow_matching/register_rcgan_full500_completion_replay_queue_v1.py',
        'tests/test_fm_rcgan_full500_completion_replay_queue_v1.py',
        'scripts/flow_matching/audit_rcgan_epoch_checkpoint_v1.py',
        'configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v4.json'}
    source_paths.update(pin['path'] for pin in queue['source_receipts'])
    routes=[]
    dispatch=read(ROOT/'configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v4.json')
    source_paths.update(pin['path'] for pin in dispatch['source_receipts'])
    for canonical in queue['jobs']:
        if canonical['seed'] in recovery_paths:
            path=recovery_paths[canonical['seed']];recovery=read(ROOT/path)
            source_paths.add(path);source_paths.update(pin['path'] for pin in recovery['source_receipts'])
            state=dispatch['state_root'] if canonical['seed']==1103 else recovery['state_root']
            route=dict(canonical_job=copy.deepcopy(canonical),execution_job=recovery['recovery_job'],kind='resume',
                recovery_configuration=path,native_receipt=state+'/receipt.json',
                successful_native_receipt_field='full_original500epoch_resume_verified')
        else:
            route=dict(canonical_job=copy.deepcopy(canonical),execution_job=copy.deepcopy(canonical),kind='original',
                recovery_configuration=None,native_receipt=queue['state_root']+'/receipts/'+canonical['id']+'.json',
                successful_native_receipt_field='full_native_result_verified')
        route['result_path']=(ROOT/route['execution_job']['output_directory']/'result.json').as_posix()
        routes.append(route)
    owners=[]
    for pid,created,fragment in [(23244,1791676903.708095,'run_rcgan_native_original_queue_v2'),
        (8168,1791737313.179449,'run_rcgan_native_state_recovery_v2'),
        (28796,1791748470.4510753,'run_rcgan_seed1103_registered_retry_dispatch_v4')]:
        process=psutil.Process(pid)
        if process.create_time()!=created or not any(fragment in part for part in process.cmdline()):
            raise ValueError('Actual original native owner identity changed; inspect before registration')
        owners.append(dict(pid=pid,create_time=created,command=process.cmdline()))
    path='configs/runtime/fm_rcgan_full500_completion_replay_queue.v1.json'
    config=dict(runtime_configuration=path,original_queue=original_path,routes=routes,
        required_epochs_per_job=500,required_generated_points_per_epoch=4984,required_generator_calls_per_epoch=178,
        full_fit_points=996,full_MMD_points=3988,equivalence_gate='bitwise',cpu_threads=2,
        python=queue['python'],settings=copy.deepcopy(queue['settings']),resource_lock=queue['resource_lock'],
        poll_seconds=10,state_root='experiments/runs/fm_rcgan_full500_completion_replay_queue_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/rcgan_full500_completion_replay_v1/completed_models',
        expected_original_native_owners=owners,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        citation='Esteban, Hyland and Raetsch. Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs. arXiv:1706.02633v2.',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(source_paths)])
    verify(config)
    exclusive(ROOT/path,config)
    evidence=ROOT/'projects/flow_matching_research/results/2026-10-12/rcgan_full500_completion_replay_v1'
    exclusive(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=sha(ROOT/path),original_scope=config['original_scope'],
        all_five_original_seeds_retained=[1103,1104,1105,1106,1107],full_epochs_per_seed=500,
        total_checkpoints_required=2500,total_generator_samples_required=12460000,
        total_generator_calls_required=445000,full_original_MMD_replays_required=2500,
        exact_two_same_seed_recovery_mappings_retained=True,actual_original_native_owners=owners,
        existing_native_training_controllers_unchanged=True,full_native_replays_executed=False,
        additional_training_seed_slots=0,new_numeric_results_published=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,original_jobs=5,total_full_checkpoints_required=2500,
        full_MMD_replays_required=2500,native_training_owners_unchanged=True)))


if __name__=='__main__':main()
