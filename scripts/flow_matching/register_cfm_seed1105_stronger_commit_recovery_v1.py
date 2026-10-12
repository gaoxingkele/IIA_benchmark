"""Preserve the unsaved1-of400 incident and raise admission without easing guards."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.run_cfm_seed1105_stronger_commit_parent_v3 import parent_namespace
from scripts.flow_matching.run_cfm_checkpointed_light_parent_v2 import ROOT

QUEUE = 'configs/experiments/fm_cfm_node1105_complete_epoch_full_retry_queue.v11.json'
CONFIG = 'configs/runtime/fm_cfm_node1105_stronger_commit_full_retry.v15.json'
REPLAY = 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
RESULT = 'projects/flow_matching_research/results/2026-10-13/cfm_seed1105_full400_stronger_commit_recovery_v1'


def exclusive(path,value):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def require_terminal(state):
    import psutil
    if state['state']!='failed_attempt_preserved' or state['active_pid'] is not None:
        raise ValueError('Only an actual terminal failed producer may be replaced')
    for pid,created in ((state['pid'],state['create_time']),(30368,1791837762.3477995),(7524,1791837762.3126793)):
        try:
            if psutil.Process(pid).create_time()==created:
                raise ValueError('Never restart or replace live scientific training')
        except psutil.NoSuchProcess:pass


def register():
    import psutil
    values=parent_namespace();original=values['original']
    read,sha,verify=original.read,original.sha,original.verify
    def pin(path):return dict(path=path,sha256=sha(ROOT/path))
    identity='cfm_ts_original__main_text__node_e400__pendulum__seed1105'
    source_path='configs/experiments/fm_cfm_ts_original_queue.v1.json'
    failed_config_path='configs/runtime/fm_cfm_node1105_complete_epoch_full_retry.v13.json'
    source=read(ROOT/source_path);verify(source)
    canonical=next(j for j in source['jobs'] if j['id']==identity)
    failed_config=read(ROOT/failed_config_path);values['verify_registration'](failed_config)
    failed_path=failed_config['queue'];failed_queue=read(ROOT/failed_path)
    failed=next(j for j in failed_queue['jobs'] if j['id']==failed_config['job_id'])
    state_path=failed_config['state_root']+'/status.json';state=read(ROOT/state_path);require_terminal(state)
    receipt_path=failed_config['state_root']+'/receipt.json'
    progress_path=failed['output_directory']+'/progress.json'
    receipt,progress=read(ROOT/receipt_path),read(ROOT/progress_path)
    if (receipt['exit_code']==0 or receipt['complete']
            or not receipt['resource_observations']['resource_guard_aborted']
            or receipt['resource_observations']['abort_resource_snapshot']['commit_available_bytes']>=16*1024**3
            or progress['epoch']!=1 or progress['optimizer_updates']!=1 or progress['required_epochs']!=400
            or values['canonical_science'](failed)!=values['canonical_science'](canonical)):
        raise ValueError('Require the actual canonical seed1105 resource-aborted1-of400 incident')
    for directory in (ROOT/failed['output_directory'],ROOT/failed['artifact_directory']):
        if directory.exists() and any(p.is_file() and p.suffix in ('.pt','.pth','.pkl') for p in directory.rglob('*')):
            raise ValueError('A saved native checkpoint requires restoration auditing before continuation')
    diagnostic_path='projects/flow_matching_research/results/2026-10-12/cfm_complete_epoch_full400_retry_v1/native_validation_v2/actual_guarded_native_checkpoint_diagnostic_receipt.json'
    diagnostic=read(ROOT/diagnostic_path)
    if diagnostic['exit_code']!=0 or not diagnostic['diagnostic_checks_passed'] or diagnostic['resource_guard_aborted']:
        raise ValueError('Successful original full-architecture checkpoint diagnostic required')
    previous_path='configs/runtime/fm_cfm_completion_replay_queue.v3.json'
    from scripts.flow_matching.run_cfm_completion_replay_queue_v2 import original as replay_source,allowed_idle
    previous=read(ROOT/previous_path);replay_source.verify(previous)
    owner_state_path=previous['state_root']+'/status.json';owner_state=read(ROOT/owner_state_path)
    actor=psutil.Process(owner_state['pid'])
    children=[dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in actor.children(recursive=True)]
    if (actor.create_time()!=owner_state['create_time'] or actor.cmdline()[-1]!=previous_path
            or not allowed_idle(owner_state,children)):
        raise ValueError('Exact idle all285 replay owner required; never stop native work')
    for path in (QUEUE,CONFIG,REPLAY,RESULT):
        if (ROOT/path).exists():raise FileExistsError('Preserve prior registrations: '+path)
    queue=deepcopy(source);job=next(j for j in queue['jobs'] if j['id']==identity)
    job.update(id=identity+'__checkpointed_full_retry_v11',recovery_original_job_id=identity,
        output_directory='experiments/runs/fm_cfm_node1105_complete_epoch_full_retry_v11/jobs/'+identity,
        artifact_directory=str(Path(failed['artifact_directory']).parent.parent/'cfm_node1105_complete_epoch_full_retry_v11'/identity))
    policy=deepcopy(failed_config['resource_policy']);policy['minimum_commit_headroom_bytes']=34359738368
    queue.update(state_root='experiments/runs/fm_cfm_node1105_complete_epoch_full_retry_v11',resource_policy=policy,
        recovery_original_queue=source_path,recovery_original_queue_sha256=sha(ROOT/source_path),
        recovery_job_ids=[job['id']],recovery_selection='Same canonical seed1105; unsaved1 epoch receives zero credit. '
        'All400 updates/full500 test points retained; admission increased22to32GiB, emergency16GiB unchanged.')
    sources=[source_path,failed_config_path,failed_path,receipt_path,progress_path,diagnostic_path,
        'scripts/flow_matching/run_cfm_seed1105_stronger_commit_parent_v3.py',
        'scripts/flow_matching/register_cfm_seed1105_stronger_commit_recovery_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_full_retry_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_light_parent_v2.py',
        'scripts/flow_matching/cfm_complete_epoch_checkpoint_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'tests/test_fm_cfm_seed1105_stronger_commit_recovery_v1.py']
    queue['source_receipts']+=[pin(p) for p in sources]
    assert len(queue['jobs'])==285
    assert all(a==b for a,b in zip(source['jobs'],queue['jobs']) if a['id']!=identity)
    assert {k:v for k,v in policy.items() if k!='minimum_commit_headroom_bytes'}=={
        k:v for k,v in failed_config['resource_policy'].items() if k!='minimum_commit_headroom_bytes'}
    exclusive(QUEUE,queue)
    config=dict(queue=QUEUE,job_id=job['id'],canonical_job=canonical,state_root=queue['state_root'],
        family_lock=failed_config['family_lock'],resource_policy=policy,checkpoint_interval=10,
        resume_full_epoch_checkpoint=None,resume_full_epoch_checkpoint_sha256=None,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        source_receipts=[pin(p) for p in sources+[QUEUE]],citation=failed_config['citation'],
        reproduction_status=failed_config['reproduction_status'],original_failed_attempt=failed,
        preserved_failed_terminal=receipt_path,discarded_unsaved1_epoch_not_counted=True)
    values['verify_registration'](config);exclusive(CONFIG,config)
    replay=deepcopy(previous);route=next(r for r in replay['routes'] if r['original_canonical_id']==identity)
    route.update(job=job,result_path=job['output_directory']+'/result.json',
        checkpointed_training_terminal=config['state_root']+'/receipt.json')
    replay.update(runtime_configuration=REPLAY,state_root='experiments/runs/fm_cfm_completion_replay_queue_v4',
        child_state_root='experiments/runs/fm_cfm_completion_replay_queue_v4/children',
        child_config_root='configs/reproducibility/fm_cfm_completion_replay_queue_v4',
        artifact_root=str(Path(previous['artifact_root']).with_name('cfm_completion_replay_queue_v4')),
        previous_owner=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline()),
        previous_state_path=owner_state_path,poll_seconds=10)
    replay['source_receipts']=[pin(p) for p in sorted({r['path'] for r in previous['source_receipts']}|set(sources+[QUEUE,CONFIG,previous_path]))]
    replay_source.verify(replay);exclusive(REPLAY,replay)
    exclusive(RESULT+'/actual_resource_abort_and_full400_registration.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),actual_terminal=receipt,actual_progress=progress,
        failed_worker=dict(pid=30368,create_time=1791837762.3477995),retained_partial_epochs_credited=0,
        original_required_epochs=400,actual_fresh_training_completed=False,checkpoint_interval=10,
        complete_epoch_checkpoints_required=40,minimum_commit_headroom_bytes=34359738368,
        previous_minimum_commit_headroom_bytes=failed_config['resource_policy']['minimum_commit_headroom_bytes'],
        emergency_commit_headroom_bytes=17179869184,source_solver_and_optimizer_unchanged=True,
        source_data_seed_and_full500_test_points_unchanged=True,canonical_job=canonical,replay_route=route,
        original_CFM_jobs=285,preserved_verified_original_models=len(owner_state['verified_original_ids']),
        other284_replay_routes_unchanged=all(a==b for a,b in zip(previous['routes'],replay['routes']) if a['original_canonical_id']!=identity),
        configuration_receipts=[pin(p) for p in (QUEUE,CONFIG,REPLAY)],previous_idle_replay_owner=replay['previous_owner'],
        original_scope=config['original_scope'],benchmark_performance=False,author_equivalence_certified=False,
        all_original_experiments_complete=False))
    print(json.dumps(dict(queue=QUEUE,configuration=CONFIG,replay=REPLAY,original_epochs=400,
        retained_partial_epochs_credited=0,minimum_commit_headroom_bytes=34359738368)))


if __name__=='__main__':register()
