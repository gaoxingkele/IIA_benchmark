"""Register three exact idle CPU requests without changing any experiment scope."""
from datetime import datetime,timezone
import json

from scripts.flow_matching.run_cached_remaining_cpu_monitors_v4 import base,verify,safe_phase
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    runtime_path='configs/runtime/fm_bounded_remaining_cpu_queues.v3.json';runtime=base.read(base.ROOT/runtime_path)
    rows=[]
    for role in ('entropic_cpu','pi_cpu','moment_cpu'):
        state_path=runtime['state_root']+'/'+role+'/status.json';state=base.read(base.ROOT/state_path)
        process=psutil.Process(state['pid']);command=process.cmdline()
        if command[command.index('-m')+1]!='scripts.flow_matching.run_bounded_remaining_cpu_queues_v3':
            raise ValueError('Exact original bounded CPU entry required')
        selected=runtime['controllers'][role];queue=base.read(base.ROOT/selected['queue'])
        job=next(j for j in queue['jobs'] if j['id']==state['requested_job'])
        path='configs/runtime/fm_'+role+'_cached_resource_monitor.v4.json'
        sources={p['path'] for p in runtime['source_receipts']}
        sources.update([runtime_path,selected['queue'],
            'scripts/flow_matching/run_cached_remaining_cpu_monitors_v4.py',
            'scripts/flow_matching/register_cached_remaining_cpu_monitors_v4.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v2.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
            'scripts/flow_matching/cached_resource_snapshot_v1.py',
            'tests/test_fm_cached_remaining_cpu_monitors_v4.py'])
        config=dict(runtime_configuration=path,entry_module='scripts.flow_matching.run_bounded_remaining_cpu_queues_v3',
            role=role,original_arguments=command[command.index('-m')+2:],original_runtime_configuration=runtime_path,
            original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
            settings=runtime['settings'],resource_lock=runtime['resource_lock'],historical_identity=None,
            pending_job=job,expected_state_identity=dict(runtime_sha256=base.sha(base.ROOT/runtime_path),
                queue_sha256=selected['queue_sha256']),
            monitor_state_root='experiments/runs/fm_'+role+'_cached_resource_monitor_v4',
            handoff=dict(previous_owner=dict(pid=process.pid,create_time=process.create_time(),command=command),
                previous_state_path=state_path),
            source_receipts=[dict(path=p,sha256=base.sha(base.ROOT/p)) for p in sorted(sources)],
            boundary='Only verified idle unstarted CPU host monitor handoff. Entire original105 entropic/1140Pi/72MOMENT '
                'registries remain bound together, selected role/native commands/validators/data/epochs/seed/thread '
                'budgets/32GiB admission/16GiB emergency/10GiB RSS/12GiB private caps remain unchanged. Pi prepared '
                'empty append-only logs may be retained; any prior model artifact blocks handoff. No frozen source '
                'or live native child is hot patched, no failed attempt retried, no performance result claimed.')
        verify(config)
        children=[dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in process.children(recursive=True)]
        if not safe_phase(config,state,children):raise ValueError('Exact original unstarted request and no native child required')
        exclusive(base.ROOT/path,config)
        output=base.ROOT/job['output_directory']
        rows.append(dict(role=role,configuration=path,configuration_sha256=base.sha(base.ROOT/path),
            previous_owner=config['handoff']['previous_owner'],previous_descendants=children,
            previous_private_bytes=process.memory_info().private,full_original_jobs=len(queue['jobs']),
            pending_job_id=job['id'],prepared_output_directory_exists=output.exists(),
            prepared_file_receipts=[dict(path=str(p),bytes=p.stat().st_size,sha256=base.sha(p)) for p in output.iterdir()] if output.exists() else []))
    evidence=base.ROOT/'projects/flow_matching_research/results/2026-10-12/cached_remaining_cpu_monitors_v4'
    exclusive(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),profiles=rows,
        original_scope=config['original_scope'],all_original_registries_budgets_and_guards_unchanged=True,
        new_benchmark_results=0,all_original_experiments_complete=False))
    print(json.dumps([dict(role=r['role'],full_jobs=r['full_original_jobs'],previous_private_bytes=r['previous_private_bytes']) for r in rows]))


if __name__=='__main__':main()
