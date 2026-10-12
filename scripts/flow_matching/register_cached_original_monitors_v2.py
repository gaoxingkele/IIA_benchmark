"""Register idle CFM and SAITS monitor handoffs while preserving every native job."""
from datetime import datetime,timezone
import json

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v2 import verify,safe_phase
    profiles=[
        ('cfm',18116,'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
         'configs/runtime/fm_cfm_ts_low_memory_controller.v3.json','experiments/runs/fm_cfm_ts_original_v1/status.json'),
        ('saits',32196,'scripts.flow_matching.run_saits_guarded_original_queue_v1',
         'configs/runtime/fm_saits_guarded_original_resume.v1.json','experiments/runs/fm_saits_guarded_original_resume_v1/status.json')]
    report=[]
    for role,pid,module,runtime_path,state_path in profiles:
        runtime=read(ROOT/runtime_path);state=read(ROOT/state_path)
        process=psutil.Process(pid);command=process.cmdline()
        original_arguments=command[command.index('-m')+2:]
        if command[command.index('-m')+1]!=module or state['pid']!=pid:
            raise ValueError('Exact original controller entry/arguments required')
        children=[dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in process.children(recursive=True)]
        path='configs/runtime/fm_'+role+'_cached_resource_monitor.v2.json'
        historical=None
        if role=='cfm':
            queue_path=runtime['controllers']['original']['queue'];queue=read(ROOT/queue_path)
            settings=queue['resource_policy'];lock=runtime['family_lock']
            proof_path='projects/flow_matching_research/results/2026-10-10/native_methods_resume_v1/runtime_observations.json'
            historical=dict(path=proof_path,sha256=sha(ROOT/proof_path),controller_name='cfm_original')
        else:
            queue_path=runtime['queue'];queue=read(ROOT/queue_path)
            settings=runtime['settings'];lock=runtime['resource_lock']
        sources={p['path'] for p in runtime['source_receipts']}
        sources.update([runtime_path,queue_path,
            'scripts/flow_matching/run_cached_resource_monitor_controller_v2.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
            'scripts/flow_matching/register_cached_original_monitors_v2.py',
            'scripts/flow_matching/cached_resource_snapshot_v1.py',
            'tests/test_fm_cached_monitor_controller_v2.py'])
        if historical: sources.add(historical['path'])
        config=dict(runtime_configuration=path,entry_module=module,original_arguments=original_arguments,
            original_runtime_configuration=runtime_path,original_scope=dict(papers=45,method_baseline_ablation_records=681,
                original_data_task_records=273),settings=settings,resource_lock=lock,
            monitor_state_root='experiments/runs/fm_'+role+'_cached_resource_monitor_v2',historical_identity=historical,
            expected_state_identity=dict(queue_sha256=sha(ROOT/queue_path),runtime_sha256=sha(ROOT/runtime_path)),
            handoff=dict(previous_owner=dict(pid=pid,create_time=process.create_time(),command=command),
                previous_state_path=state_path),
            source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)],
            boundary='Only verified idle host monitor handoff. Full original native controller main/arguments/queue '
                'and every validator, data/model field, training budget, seed, GPU/CPU guard and output path retained. '
                'Windows telemetry reuses equivalent ctypes/API objects. No live native process or frozen source '
                'is hot patched. Runtime status gains exact creation time metadata only. Partial stage state is '
                'not fabricated; active CrossAD and prepared MaelNet pipelines are excluded from this handoff.')
        verify(config)
        if not safe_phase(config,state,children):
            raise ValueError('Only independently verified idle owner without native children may be handed off')
        exclusive(ROOT/path,config)
        report.append(dict(role=role,configuration=path,configuration_sha256=sha(ROOT/path),
            previous_owner=config['handoff']['previous_owner'],previous_descendants=children,
            previous_private_bytes=process.memory_info().private,original_jobs=len(queue['jobs'])))
    base=ROOT/'projects/flow_matching_research/results/2026-10-12/cached_original_monitors_v2'
    exclusive(base/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),profiles=report,
        original_scope=config['original_scope'],original_caps_and_native_queues_unchanged=True,
        active_CrossAD_and_prepared_MaelNet_owners_excluded=True,new_benchmark_results=0,
        all_original_experiments_complete=False))
    print(json.dumps([dict(role=r['role'],original_jobs=r['original_jobs'],private_bytes=r['previous_private_bytes']) for r in report]))


if __name__=='__main__': main()
