"""Freeze actual failures, full-budget path/recomputation adapter and native CUDA probes."""
from datetime import datetime, timezone
import psutil
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.register_flow_mismatching_full_image_evaluation_v1 import create, receipt


def main():
    cases=[];failures=[]
    target='projects/flow_matching_research/results/2026-10-11/full_image_execution_recovery_v2'
    for family,objective in [('fm_wt_full_image_training_v1','WT'),('fm_flow_mismatching_full_training_v1','FM')]:
        state_path='experiments/runs/'+family+'/status.json';state=read(ROOT/state_path)
        if (psutil.pid_exists(state['pid']) or state['state']!='failed_attempt_preserved'
                or state['last_terminal']['returncode']!=1):
            raise ValueError('Authoritative missing original controller and preserved native terminal failure required')
        command=state['last_terminal']['command'];job_path=command[command.index('--config')+1]
        job=read(ROOT/job_path);verify_receipts(job['source_receipts'])
        attempt=command[command.index('--attempt-directory')+1]
        if (ROOT/job['output_directory']/'resume.pt').exists() or (ROOT/job['output_directory']/'training_completed.json').exists():
            raise ValueError('These original failures must precede any complete training epoch')
        error_path=attempt+'/stderr.log';terminal_path=attempt+'/terminal.json'
        failure=dict(original_state=receipt(state_path),original_attempt_terminal=receipt(terminal_path),
            original_stderr=receipt(error_path),actual_terminal_state=state,
            original_training_configuration=receipt(job_path),full_original_training_complete=False)
        if objective=='FM' and 'FileNotFoundError' not in (ROOT/error_path).read_text(encoding='utf-8'):
            raise ValueError('Actual preserved Windows temporary path failure required')
        if objective=='WT' and not state['last_terminal']['resource_usage']['resource_guard_aborted']:
            raise ValueError('Actual preserved full-batch original GPU capacity guard failure required')
        failures.append(failure)
        create(target+'/'+objective.lower()+'_original_terminal_failure.json',failure)
        cases.append(dict(id='full_native_'+objective.lower()+'_batch32_seed1103',objective=objective,
            original_training_configuration=job_path,
            output_directory='data/public_datasets/flow_matching/experiments/full_image_execution_recovery_v2/capacity/'+objective.lower()))
    sources=['scripts/flow_matching/full_image_training_execution_recovery_v2.py',
        'scripts/flow_matching/probe_full_image_cuda_execution_recovery_v2.py',
        'scripts/flow_matching/register_full_image_execution_recovery_v2.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/wt_full_image_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_v1.py',
        'scripts/flow_matching/run_wt_full_image_training_v1.py',
        'src/iia_benchmark/models/fm_supplemental_author.py',
        'src/iia_benchmark/models/fm_optimizer_reference.py','src/iia_benchmark/models/fm_wt_image_training.py',
        'src/iia_benchmark/models/fm_reflect_wt_native.py','tests/test_fm_full_image_execution_recovery_v2.py']
    sources += [case['original_training_configuration'] for case in cases]
    config_path='configs/reproducibility/fm_full_image_execution_recovery.2026-10-11_v2.json'
    config=dict(schema_version=1,capacity_cases=cases,original_failures=failures,
        execution_adapter=dict(Windows_native_extended_paths=True,activation_recomputation=True,
            use_reentrant=False,preserve_rng_state=True,native_ResBlocks=54,native_AttentionBlocks=29,
            model_parameters=45289539,original_batch_size=32,original_main_and_WT_epochs=2000,
            smaller_batches_or_models_allowed=False,source_files_modified=False),
        checkpoint_documentation='https://docs.pytorch.org/docs/2.14/checkpoint.html',
        installed_training_Torch_version_remains_original_frozen_2_13=True,
        source_receipts=[receipt(path) for path in sources],
        original_scope=read(ROOT/'configs/experiments/fm_wt_full_image_training_queue.v1.json')['original_scope'],
        full_original_training_restarted=False,full_original_training_completed=False,
        benchmark_performance=False,all_original_experiments_complete=False,
        boundary='Capacity-only full original32batch/backbone/optimizer update and checkpoint IO. Complete training recovery requires successful capacity proof and separate exact-budget registration; no shortened reproduction or performance claim.')
    created=[create(config_path,config)]
    parent=read(ROOT/'configs/runtime/fm_wt_full_image_training_queue.v1.json')
    runtime_path='configs/runtime/fm_full_image_cuda_capacity_queue.v2.json'
    runtime_sources=[config_path,'scripts/flow_matching/run_full_image_cuda_capacity_queue_v2.py',
        'scripts/flow_matching/run_tccm_full_image_evaluation_queue_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_queue_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py']
    created.append(create(runtime_path,dict(schema_version=1,configuration=config_path,python=parent['python'],
        resource_lock=parent['resource_lock'],settings=parent['settings'],
        lane_lock='experiments/runs/fm_full_image_cuda_capacity_v2/queue.lock',
        state_root='experiments/runs/fm_full_image_cuda_capacity_v2',source_receipts=[receipt(path) for path in runtime_sources])))
    created.append(create(target+'/registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=receipt(config_path),runtime=receipt(runtime_path),created_files=created.copy(),
        required_capacity_cases=2,native_batch_size=32,full_model_parameters=45289539,
        original_scope=config['original_scope'],native_original_full_training_complete=False,new_measured_seed_slots=0,
        all_original_experiments_complete=False)))
    print(dict(original_failures_preserved=2,full_native_batch32_capacity_cases=2,training_budget_not_reduced=True))


if __name__=='__main__':main()
