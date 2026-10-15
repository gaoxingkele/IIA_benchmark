"""Freeze every original/repair/preprocessing objective case and failure record."""
import json
from scripts.flow_matching import run_objective_full440_checkpoint_replay_v1 as runner

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
CONFIG='configs/runtime/fm_objective_full440_checkpoint_replay.v1.json'


def registration():
    parent=read(ROOT/runner.RESOURCE);old=read(ROOT/runner.OLD_SB)
    files=[runner.RESOURCE,runner.LEGACY_TERMINAL,runner.OLD_SB,
        old['output_directory']+'/independent_full_model_score_verification.json',old['state_root']+'/receipt.json',
        'scripts/flow_matching/objective_replay_math_v1.py',
        'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
        'scripts/flow_matching/register_objective_full440_checkpoint_replay_v1.py',
        'tests/test_fm_objective_full440_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_tsad_queue.py']
    identities=[]
    for path,_ in runner.QUEUES.values():
        files.append(path)
        for job in read(ROOT/path)['jobs']:
            identities.append(job['id']);files.append(job['model_config'])
            files.extend(p['path'] for p in job['frozen_files'] if p['path'].endswith(('.py','.json')))
    terminal=read(ROOT/runner.LEGACY_TERMINAL)
    return dict(schema_version=1,queues={k:dict(path=p,count=n) for k,(p,n) in runner.QUEUES.items()},
        required_cases=440,required_ids=identities,settings=parent['settings'],resource_lock=parent['resource_lock'],
        controller_python=parent['python'],math_tolerance=dict(rtol=2e-6,atol=2e-6),array_comparison='exact_dtype_shape_content_bytes',
        probability_path_pass_criterion='independent_numpy_source_FP32_operation_program',
        ideal_float64_path_audit='all_coordinate_failure_counts_and_errors_retained_diagnostic',
        preserved_original_failures={r['id']:r for r in terminal['records'] if r['status']=='failed' and r.get('returncode')!=0},
        existing_native_certificate=dict(configuration=runner.OLD_SB,id=old['job_id']),
        source_guard_directories=dict(
            original=['experiments/runs/fm_resource_gated_cpu_resume_v1/strict_cpu/receipts'],
            repair=['experiments/runs/fm_resource_gated_cpu_resume_v1/entropic_cpu/receipts','experiments/runs/fm_bounded_remaining_cpu_v3/entropic_cpu/receipts'],
            constant_fit=['experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/cpu/receipts'],
            constant_fit_repair=['experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/entropic_numeric_repair/receipts']),
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        artifact_root='F:/aicoding/IIA_Data/experiments/objective_full440_checkpoint_replay_v1',
        state_root='experiments/runs/fm_objective_full440_checkpoint_replay_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        boundary='All280original+105numerical repairs+40constant-fit+15constant-fit repairs kept separate. '
            'Original105failed numerical attempts are preserved negative outcomes,not automatically retried '
            'or successful models. Completed models restore every trained tensor,including optional SF2M '
            'score head,without fitting;score calls retain seed+1,batch64,all4MCsamples x5times,all100point '
            'windows/end-aligned tails and4CPUthreads. Full scores/labels match dtype/shape/content bytes. '
            'Every fresh native FP32path-program coordinate and float64MCresidual is independently checked '
            'at fixed2e-6tolerance. Ideal float64path comparisons keep all failure coordinate counts/errors; '
            'FP32cancellation differences are disclosed,not recertified as high-precision equivalence. '
            'Separate child replays original metrics/events/200resample intervals and '
            'independently checks validation thresholds,confusion/ranking,PA and ten test-pooled controls. '
            'The alreadyclosedSB_SWaTseed1103 native proof is reused for a new independent scalar review; '
            'no fresh inlinepath audit is falsely attributed to that reused native proof. '
            'Missing original execution-guard history stays unproven. No preprocessing or VUS/Affiliation '
            'integral reconstruction,no author image/single-cell/TSAD or fullTAB equivalence, '
            'new training or seed. Full16/32/16GiBprotection and10/12GiBmemorycaps remain unchanged.')


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Preserve frozen all440objective case registration')
    config=registration();runner.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(config,f,indent=2);f.write('\n')
    print(json.dumps(dict(configuration=CONFIG,cases=440,preserved_original_failure_outcomes=105,reused_native_models=1,new_training_jobs=0)))


if __name__=='__main__':main()
