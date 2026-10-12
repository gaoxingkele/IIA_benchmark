from scripts.flow_matching.run_cfm_ts_original_job import ROOT,read
from scripts.flow_matching.register_cfm_complete_epoch_full400_retry_v1 import QUEUE,CONFIG,REPLAY
from scripts.flow_matching.cfm_complete_epoch_checkpoint_v1 import canonical_science


def test_registered_producer_preserves_every285_case_and_full_source_budget():
    queue=read(ROOT/QUEUE);config=read(ROOT/CONFIG)
    original=read(ROOT/'configs/experiments/fm_cfm_ts_original_queue.v1.json')
    assert len(queue['jobs'])==len(original['jobs'])==285
    canonical={j['id']:j for j in original['jobs']}
    for job in queue['jobs']:
        name=job.get('recovery_original_job_id',job['id'])
        assert canonical_science(job)==canonical_science(canonical[name])
    selected=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    assert selected['epochs']==selected['expected_optimizer_updates']==400
    assert selected['seed']==1105 and selected['batch_size']==258 and selected['test_points_per_trajectory']==500
    assert config['resume_full_epoch_checkpoint'] is None
    assert config['checkpoint_interval']==10 and config['discarded_unsaved308_epochs_not_counted']
    assert queue['resource_policy']==config['resource_policy']==original['resource_policy']
    assert config['reproduction_status']=='executable_paper_derived_reconstruction_not_author_equivalent'


def test_registered_replay_preserves_all285_slots_and_existing85_proofs():
    old=read(ROOT/'configs/runtime/fm_cfm_completion_replay_queue.v1.json');new=read(ROOT/REPLAY)
    assert len(new['routes'])==285
    assert [r['original_canonical_id'] for r in new['routes']]==[r['original_canonical_id'] for r in old['routes']]
    assert new['preserved_verified_union']==old['preserved_verified_union']
    assert sum(a!=b for a,b in zip(old['routes'],new['routes']))==1
    for key in ('settings','resource_lock','prediction_tolerance','environment_lock','native_replay_template','original_scope'):
        assert new[key]==old[key]


def test_actual_handoff_preserves85_certificate_ids_and_stops_no_native_child():
    config=read(ROOT/REPLAY)
    handoff=read(ROOT/config['state_root']/'handoff.json')
    old=read(ROOT/config['state_root']/'previous_status_at_verified_handoff.json')
    assert handoff['no_native_child_present'] and handoff['waiting_controller_only_stopped']
    assert not handoff['original_native_child_stopped']
    assert old['active_pid'] is None and len(old['verified_original_ids'])==85 and not old['failed_preserved']
    state=read(ROOT/config['state_root']/'status.json')
    assert state['required_original_models']==285
    assert set(old['verified_original_ids']).issubset(state['verified_original_ids'])
