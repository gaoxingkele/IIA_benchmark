"""Register78 source-preserving complete MOMENT native replay/review jobs."""
import json
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as runner

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
CONFIG='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'


def registration():
    original=read(ROOT/runner.RESOURCE);older='configs/runtime/fm_resource_gated_cpu_resume.v1.json'
    files=[runner.RESOURCE,older,*[v[0] for v in runner.QUEUES.values()],
        'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/register_moment_full_checkpoint_replay_v1.py',
        'tests/test_fm_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_tsad_queue.py']
    ids=[]
    for path,_ in runner.QUEUES.values():
        for job in read(ROOT/path)['jobs']:
            ids.append(job['id']);files.append(job['model_config'])
            files.extend(pin['path'] for pin in job['frozen_files'] if pin['path'].endswith(('.py','.json')))
    return dict(schema_version=1,queues={k:dict(path=p,count=n) for k,(p,n) in runner.QUEUES.items()},
        required_jobs=78,required_ids=ids,settings=original['settings'],resource_lock=original['resource_lock'],
        controller_python=original['python'],source_guard_directories=dict(
            original=[read(ROOT/older)['state_root']+'/moment_cpu/receipts',original['state_root']+'/moment_cpu/receipts'],
            constant_fit=['experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/moment_pretrained/receipts']),
        array_comparison='exact_dtype_shape_content_bytes',mse_tolerance=dict(rtol=2e-6,atol=2e-6),
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_full78_checkpoint_replay_v1',
        state_root='experiments/runs/fm_moment_full78_checkpoint_replay_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        boundary='All72 original small/base/large,context512/100 and12dataset jobs,plus all6 SMAP constant-fit '
            'controls retained separately. Fresh exact producer-environment CPU inference from full registered '
            'official weights for every train/validation/test group and timestamp,batch1 and all-observed masks. '
            'Every saved score,label,timestamp,source-row and fault-label array must match dtype/shape/bytes. '
            'Independent float64 MSE checked on every full forward window. Separate child reviews all arrays, '
            'replays complete original metrics/events and200resample grouped intervals,and independently checks '
            'validation thresholds,confusion/rank/PA and ten test-pooled diagnostic ratios. '
            'Raw preprocessing and VUS/Affiliation integrals are not independently reconstructed here. '
            'Missing historical source guard evidence stays unproven; fresh guards cannot retroactively certify it. '
            'No fine-tuning,new training seed,retry,download,historical-paper equivalence or fullTAB certification. '
            'Original16GiBphysical/32GiBcommit/16GiBemergency and10GiBRSS/12GiBprivate caps retained unchanged.')


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Preserve frozen full78 MOMENT replay registration')
    runtime=registration();runner.verify(runtime)
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(runtime,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG,original_jobs=72,additional_preprocessing_controls=6,unique_replay_jobs=78,new_training_seeds=0)))


if __name__=='__main__':main()
