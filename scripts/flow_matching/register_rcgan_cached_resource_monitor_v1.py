"""Freeze an idle-only full500 RCGAN host monitor handoff with identical caps."""
from datetime import datetime, timezone
import json

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    queue_path='configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json'
    queue=read(ROOT/queue_path)
    state_path=queue['state_root']+'/status.json'; state=read(ROOT/state_path)
    process=psutil.Process(state['pid'])
    command=process.cmdline()
    children=[dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in process.children(recursive=True)]
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import handoff_phase_allowed
    if (process.create_time()!=state['create_time'] or command[-1]!=queue_path
            or 'scripts.flow_matching.run_rcgan_native_original_queue_v2' not in command
            or not handoff_phase_allowed('original',state,False,children)):
        raise ValueError('Exact idle full500 controller without any native descendants required')
    path='configs/runtime/fm_rcgan_cached_resource_monitor.v1.json'
    sources={p['path'] for p in queue['source_receipts']}
    sources.update([queue_path,'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/register_rcgan_cached_resource_monitor_v1.py',
        'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_rcgan_native_original_queue_v2.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
        'tests/test_fm_cached_resource_snapshot_v1.py'])
    config=dict(entry_module='scripts.flow_matching.run_rcgan_native_original_queue_v2',original_queue=queue_path,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        settings=queue['settings'],resource_lock=queue['resource_lock'],
        monitor_state_root='experiments/runs/fm_rcgan_cached_resource_monitor_v1',
        handoff=dict(previous_owner=dict(pid=process.pid,create_time=process.create_time(),command=command),
            previous_state_path=state_path,runtime_configuration=path),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)],
        boundary='Only verified idle host monitor handoff. Same5 canonical full500-epoch native training jobs, '
            'original native worker module/queue/receipts/outputs and every guard retained. Startup binds only '
            'freshly compiled Windows resource_snapshot to reusable equivalent ctypes/API types. No live native '
            'process or frozen source is patched. No experiment completion or benchmark score is claimed.')
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import verify
    verify(config)
    exclusive(ROOT/path,config)
    base=ROOT/'projects/flow_matching_research/results/2026-10-12/cached_resource_monitor_v1'
    exclusive(base/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=sha(ROOT/path),original_scope=config['original_scope'],
        actual_previous_idle_owner=config['handoff']['previous_owner'],actual_previous_descendants=children,
        original_guards=config['settings'],original5_seed_full500_budget_retained=True,
        native_dispatch_and_completion_receipt_paths_unchanged=True,
        new_benchmark_results=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,previous_idle_pid=process.pid,
        previous_idle_private_bytes=process.memory_info().private,original_jobs=5,original_epochs=500)))


if __name__=='__main__': main()
