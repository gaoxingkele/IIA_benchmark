"""Preserve all285 CFM cases while recovering the lost400-epoch seed1105 slot."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching.run_cfm_ts_original_job import ROOT,read,sha,verify


QUEUE='configs/experiments/fm_cfm_node1105_complete_epoch_full_retry_queue.v9.json'
CONFIG='configs/runtime/fm_cfm_node1105_complete_epoch_full_retry.v13.json'
REPLAY='configs/runtime/fm_cfm_completion_replay_queue.v2.json'
RESULT='projects/flow_matching_research/results/2026-10-12/cfm_complete_epoch_full400_retry_v1'


def exclusive(path,value):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def pin(path):
    return dict(path=str(path),sha256=sha(ROOT/path))


def register():
    import psutil
    from scripts.flow_matching.register_cfm_ts_exact_recovery_v1 import validate_recovery
    from scripts.flow_matching.run_cfm_checkpointed_original_job_v1 import verify_registration
    from scripts.flow_matching.run_cfm_completion_replay_queue_v2 import original as old_replay,allowed_idle
    diagnostic=read(ROOT/RESULT/'native_validation_v2/actual_guarded_native_checkpoint_diagnostic_receipt.json')
    if diagnostic['exit_code']!=0 or not diagnostic['diagnostic_checks_passed'] or diagnostic['resource_guard_aborted']:
        raise ValueError('Full-architecture native checkpoint diagnostics must pass unchanged gates before registration')
    original_path='configs/experiments/fm_cfm_ts_original_queue.v1.json'
    source=read(ROOT/original_path);verify(source)
    identity='cfm_ts_original__main_text__node_e400__pendulum__seed1105'
    canonical=next(j for j in source['jobs'] if j['id']==identity)
    failed_queue_path='configs/experiments/fm_cfm_ts_exact_recovery_queue.v6.json'
    failed_queue=read(ROOT/failed_queue_path)
    failed_job=next(j for j in failed_queue['jobs'] if j['id']==identity+'__exact_recovery_v5')
    receipt_path=failed_queue['state_root']+'/receipts/'+failed_job['id']+'.json'
    progress_path=failed_job['output_directory']+'/progress.json'
    receipt,progress=read(ROOT/receipt_path),read(ROOT/progress_path)
    if (receipt['exit_code']==0 or receipt['complete'] or not receipt['resource_observations']['resource_guard_aborted']
            or progress['epoch']!=308 or progress['required_epochs']!=400 or progress['optimizer_updates']!=308):
        raise ValueError('Require the exact authoritative incomplete308-of400 incident')
    for directory in [ROOT/failed_job['output_directory'],ROOT/failed_job['artifact_directory']]:
        if directory.exists() and any(p.is_file() and p.suffix in ('.pt','.pth','.pkl') for p in directory.rglob('*')):
            raise ValueError('Any actual saved model/optimizer needs state auditing instead of discarding it')
    for process in psutil.process_iter(['pid','create_time','cmdline']):
        try:
            command=process.info['cmdline'] or []
            if ('scripts.flow_matching.run_cfm_ts_original_job' in command
                    and any(value.startswith(identity) for value in command)):
                raise ValueError('Never register duplicate live scientific work for this canonical slot')
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    for path in (QUEUE,CONFIG,REPLAY):
        if (ROOT/path).exists():raise FileExistsError('Preserve earlier full registrations: '+path)
    queue=deepcopy(source)
    job=next(j for j in queue['jobs'] if j['id']==identity)
    job.update(id=identity+'__checkpointed_full_retry_v9',recovery_original_job_id=identity,
        output_directory='experiments/runs/fm_cfm_node1105_complete_epoch_full_retry_v9/jobs/'+identity,
        artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_node1105_complete_epoch_full_retry_v9/'+identity)
    queue.update(state_root='experiments/runs/fm_cfm_node1105_complete_epoch_full_retry_v9',
        recovery_original_queue=original_path,recovery_original_queue_sha256=sha(ROOT/original_path),
        recovery_job_ids=[job['id']],recovery_selection='Same original canonical seed/full400 budget; unsaved308 epochs get no credit. No metric-dependent selection.',
        boundary=source['boundary']+' Complete epoch model/Adam/gradient/CPU NumPy Python RNG/history/order observations every10 epochs. Original solver, data, all500 test points, source guards and paper-derived rather than author-equivalent status unchanged.')
    new_sources=['scripts/flow_matching/'+n for n in (
        'cfm_complete_epoch_checkpoint_v1.py','run_cfm_checkpointed_original_job_v1.py',
        'run_cfm_checkpointed_full_retry_v1.py','run_cfm_completion_replay_queue_v2.py',
        'register_cfm_complete_epoch_full400_retry_v1.py','cached_resource_snapshot_v1.py')]
    new_sources+=['tests/'+n for n in ('test_fm_cfm_complete_epoch_source_parity_v1.py',
        'test_fm_cfm_complete_epoch_native_state_v1.py','test_fm_cfm_full285_stateful_route_v2.py')]
    new_sources+=[original_path,failed_queue_path,receipt_path,progress_path,
        RESULT+'/native_validation_v2/actual_guarded_native_checkpoint_diagnostic_receipt.json']
    queue['source_receipts']+=[pin(p) for p in new_sources]
    validate_recovery(source,queue);exclusive(QUEUE,queue)
    config=dict(queue=QUEUE,job_id=job['id'],canonical_job=canonical,state_root=queue['state_root'],
        family_lock='experiments/runs/fm_cfm_ts_original_cpu.lock',resource_policy=queue['resource_policy'],
        checkpoint_interval=10,resume_full_epoch_checkpoint=None,resume_full_epoch_checkpoint_sha256=None,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        source_receipts=[pin(p) for p in new_sources+[QUEUE]],
        citation=read(ROOT/source['protocol'])['citation'],
        reproduction_status='executable_paper_derived_reconstruction_not_author_equivalent',
        original_failed_attempt=failed_job,preserved_failed_terminal=receipt_path,
        discarded_unsaved308_epochs_not_counted=True)
    verify_registration(config);exclusive(CONFIG,config)
    previous_path='configs/runtime/fm_cfm_completion_replay_queue.v1.json'
    previous=read(ROOT/previous_path);old_replay.verify(previous)
    state_path=previous['state_root']+'/status.json';state=read(ROOT/state_path)
    actor=psutil.Process(state['pid']);command=actor.cmdline()
    children=[dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in actor.children(recursive=True)]
    preserved_ids=sorted(r['original_canonical_id'] for r in read(ROOT/previous['preserved_verified_union'])['records'])
    if (actor.create_time()!=state['create_time'] or command[-1]!=previous_path
            or 'scripts.flow_matching.run_cfm_completion_replay_queue_v1' not in command
            or not allowed_idle(state,children) or sorted(state['verified_original_ids'])!=preserved_ids):
        raise ValueError('Require the exact idle all285 owner with precisely85 existing certificates and no native child')
    replay=deepcopy(previous)
    route=next(r for r in replay['routes'] if r['original_canonical_id']==identity)
    route.update(job=job,result_path=job['output_directory']+'/result.json',
                 checkpointed_training_terminal=config['state_root']+'/receipt.json')
    replay.update(runtime_configuration=REPLAY,state_root='experiments/runs/fm_cfm_completion_replay_queue_v2',
        child_state_root='experiments/runs/fm_cfm_completion_replay_queue_v2/children',
        child_config_root='configs/reproducibility/fm_cfm_completion_replay_queue_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/cfm_completion_replay_queue_v2',
        previous_owner=dict(pid=actor.pid,create_time=actor.create_time(),command=command),
        previous_state_path=state_path,poll_seconds=10)
    replay_sources={p['path'] for p in previous['source_receipts']}
    replay_sources.update(new_sources+[QUEUE,CONFIG,previous_path])
    replay['source_receipts']=[pin(p) for p in sorted(replay_sources)]
    replay['boundary']+=' Same85 existing independent certificates are reused, all284 other routes remain identical. Only failed seed1105 now points to full400 checkpointed retry, with actual successful producer terminal required. Original native grid/model and independent review workers and frozen tolerance/caps remain unchanged.'
    old_replay.verify(replay);exclusive(REPLAY,replay)
    record=dict(captured_utc=datetime.now(timezone.utc).isoformat(),queue=QUEUE,configuration=CONFIG,replay=REPLAY,
        original_scope=config['original_scope'],original_CFM_cases=285,preserved_independent_verified_cases=85,
        other284_replay_routes_unchanged=all(a==b for a,b in zip(previous['routes'],replay['routes']) if a['original_canonical_id']!=identity),
        full_original_epochs=400,original_optimizer_updates=400,full_test_points=500,checkpoint_interval=10,
        expected_complete_epoch_checkpoints=40,unsaved308_epochs_not_credited=True,
        source_solver_and_optimizer_code_unchanged=True,source_guards_and_prediction_tolerance_unchanged=True,
        native_checkpoint_diagnostic_receipt_sha256=sha(ROOT/RESULT/'native_validation_v2/actual_guarded_native_checkpoint_diagnostic_receipt.json'),
        native_diagnostics_are_not_original_experiment_results=True,
        previous_idle_replay_owner=replay['previous_owner'],previous_descendants=children,
        actual_full400_retry_completed=False,author_equivalence_certified=False,all_original_experiments_complete=False)
    exclusive(RESULT+'/actual_full400_checkpointed_retry_and_full285_replay_registration.json',record)
    return record


if __name__=='__main__':print(register())
