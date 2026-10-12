"""Register exact four unstarted180-job GPU owners and all completed source proofs."""
import json
from pathlib import Path
import psutil

from scripts.flow_matching import run_cached_idle_spectral_gpu_monitors_v1 as monitor
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    root=monitor.base.ROOT; rows=monitor.base.read(root/'tmp/idle_large_gpu_owner_bindings_20261012_v1.json')
    for row in rows:
        role=row['role']; queue=monitor.base.read(root/row['queue']); state=monitor.base.read(root/row['state_path'])
        process=psutil.Process(row['owner']['pid'])
        if process.create_time()!=row['owner']['create_time'] or process.cmdline()!=row['owner']['command']:
            raise ValueError('Exact previously observed original idle owner required')
        receipts=[]
        if role!='ls4_ushcn':
            module=__import__(monitor.ROLES[role][0],fromlist=['proof_ready'])
            for name in (('integration','data_audit') if role=='sashimi' else ('data_audit',)):
                if not module.proof_ready(queue,name): raise ValueError('Every complete earlier original data/evaluator proof required')
                path=queue[name+'_output']; receipts.append(dict(path=path,sha256=monitor.base.sha(root/path)))
        logs=root/queue['state_root']/'logs'; absent=[queue['gpu_preflight_artifact'],queue['gpu_preflight_output']]
        if role=='ls4_ushcn':
            absent.extend((logs/('gpu_preflight__'+case['id']+'.stdout.log')).relative_to(root).as_posix() for case in queue['capacity_cases'])
        else: absent.append((logs/'gpu_preflight.stdout.log').relative_to(root).as_posix())
        config_path='configs/runtime/fm_'+role+'_cached_idle_gpu_monitor.v1.json'
        files=[row['queue'],'scripts/flow_matching/run_cached_idle_spectral_gpu_monitors_v1.py',
            'scripts/flow_matching/register_cached_idle_spectral_gpu_monitors_v1.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v2.py',
            'scripts/flow_matching/cached_resource_snapshot_v1.py','scripts/flow_matching/run_spectral_original_queue.py',
            'scripts/flow_matching/full_gpu_memory_slot_v2.py']
        cfg=dict(runtime_configuration=config_path,role=role,entry_module=monitor.ROLES[role][0],
            original_runtime_configuration=row['queue'],original_arguments=['--queue',row['queue']],
            original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
            settings=queue['settings'],resource_lock=queue['resource_lock'],historical_identity=None,
            expected_state_identity=dict(queue_sha256=monitor.base.sha(root/row['queue'])),
            monitor_state_root='experiments/runs/fm_'+role+'_cached_idle_gpu_monitor_v1',
            handoff=dict(previous_owner=row['owner'],previous_state_path=row['state_path']),
            completed_diagnostic_receipts=receipts,completed_log_snapshot={p.name:monitor.base.sha(p) for p in logs.glob('*') if p.is_file()},
            original_log_directory=logs.relative_to(root).as_posix(),pending_paths_that_must_remain_absent=absent,
            source_receipts=[dict(path=p,sha256=monitor.base.sha(root/p)) for p in files],
            boundary='Exact full original GPU pipeline/queue/source/evaluator and every prior complete diagnostic '
                'retained. Only a proven idle native-child-free parent is replaced. All commit/physical/GPU '
                'thresholds, CPU thread count, source500epoch budgets and35second fairness yield unchanged. '
                'Original native child code is never hot patched. Source-case limitations remain explicit.')
        monitor.verify(cfg)
        if not monitor.safe_phase(cfg,state,[dict(name=p.name(),exe=p.exe()) for p in process.children(recursive=True)]):
            raise ValueError('Only exact genuinely unstarted original GPU preflight may be handed off')
        exclusive(root/config_path,cfg)
        print(json.dumps(dict(role=role,full_original_jobs=len(queue['jobs']),previous_pid=process.pid,
            completed_source_diagnostics_preserved=len(receipts))))


if __name__=='__main__':main()
