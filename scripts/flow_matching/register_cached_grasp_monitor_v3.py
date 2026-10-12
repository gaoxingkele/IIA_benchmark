"""Freeze idle-only GRASP host telemetry migration, preserving the full protocol."""
from datetime import datetime,timezone
import json

from scripts.flow_matching.run_cached_grasp_monitor_v3 import base,verify,safe_phase
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    runtime_path='configs/runtime/fm_grasp_full_gpu_memory_fairness.v6.json'
    runtime=base.read(base.ROOT/runtime_path);queue=base.read(base.ROOT/runtime['queue'])
    state_path=queue['state_root']+'/status.json';state=base.read(base.ROOT/state_path)
    process=psutil.Process(state['pid']);command=process.cmdline()
    if command[command.index('-m')+1]!='scripts.flow_matching.run_grasp_gpu_memory_fair_queue_v6':
        raise ValueError('Exact original full GRASP host owner required')
    proof_path='projects/flow_matching_research/results/2026-10-11/grasp_gpu_and_full_audit_resume_v2/runtime_observations.json'
    path='configs/runtime/fm_grasp_cached_resource_monitor.v3.json'
    sources={p['path'] for p in runtime['source_receipts']}
    sources.update([runtime_path,proof_path,'scripts/flow_matching/run_cached_grasp_monitor_v3.py',
        'scripts/flow_matching/register_cached_grasp_monitor_v3.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v2.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py','tests/test_fm_cached_grasp_monitor_v3.py'])
    config=dict(runtime_configuration=path,entry_module='scripts.flow_matching.run_grasp_gpu_memory_fair_queue_v6',
        original_runtime_configuration=runtime_path,original_arguments=command[command.index('-m')+2:],
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        settings=queue['settings'],resource_lock=queue['resource_lock'],
        monitor_state_root='experiments/runs/fm_grasp_cached_resource_monitor_v3',
        expected_state_identity=dict(queue_sha256=runtime['queue_sha256']),
        historical_identity=dict(path=proof_path,sha256=base.sha(base.ROOT/proof_path),
            controller_name='grasp_full_gpu_memory_v6'),
        handoff=dict(previous_owner=dict(pid=process.pid,create_time=process.create_time(),command=command),
            previous_state_path=state_path),
        source_receipts=[dict(path=p,sha256=base.sha(base.ROOT/p)) for p in sorted(sources)],
        boundary='Only independently identified idle full GRASP host monitor is handed off. All9120 original '
            'entity jobs/480cohorts/10seeds/12profiles/full1500epochs/score sensitivities and original35second '
            'fairness yield, GPU identity/VRAM/commit/emergency guards and native CUDA worker remain unchanged. '
            'Fresh orchestration telemetry reuses equivalent Windows ctypes/API objects; runtime status adds '
            'creation time only. No active native child is modified or stopped; no performance completion claimed.')
    verify(config)
    children=[dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in process.children(recursive=True)]
    if not safe_phase(config,state,children):raise ValueError('Verified idle owner with no native child required')
    exclusive(base.ROOT/path,config)
    evidence=base.ROOT/'projects/flow_matching_research/results/2026-10-12/cached_grasp_monitor_v3'
    exclusive(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=base.sha(base.ROOT/path),original_scope=config['original_scope'],
        actual_previous_owner=config['handoff']['previous_owner'],actual_previous_descendants=children,
        actual_previous_private_bytes=process.memory_info().private,full_original_jobs=9120,full_original_cohorts=480,
        native_worker_all_budgets_and_guards_unchanged=True,new_benchmark_results=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,previous_idle_pid=process.pid,original_jobs=9120,
        previous_private_bytes=process.memory_info().private)))


if __name__=='__main__':main()
