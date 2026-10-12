from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as original

ORIGINAL='configs/runtime/fm_crossad_completion_post_API_fix.v2.json'
RECOVERY='configs/runtime/fm_crossad_recovery_completion_post_API_fix.v2.json'
HANDOFF='configs/runtime/fm_crossad_full139_post_API_fix_handoff.v2.json'


def test_entire_original139_science_recipes_seeds_guards_and_prior_stage_paths_identical():
    old=original.read('configs/runtime/fm_crossad_completion_replay_queue.v1.json');new=original.read(ORIGINAL)
    assert {k:v for k,v in old.items() if k!='source_receipts'}=={k:v for k,v in new.items() if k!='source_receipts'}
    jobs=original.validate_scope(original.read(new['original_queue']),new)
    assert len(jobs)==139 and sum(j['mode']=='fresh_train' for j in jobs.values())==132
    assert sum('ablation_row' in j for j in jobs.values())==90
    assert len(new['preserved_verified_jobs'])==3


def test_exact_four_recovery_jobs_and_all_output_data_and_environment_bindings_preserved():
    old=original.read('configs/runtime/fm_crossad_recovery_completion_handoff.v1.json');new=original.read(RECOVERY)
    allowed={'source_receipts','original_completion_configuration','external_completed_recoveries'}
    assert {k:v for k,v in old.items() if k not in allowed}=={k:v for k,v in new.items() if k not in allowed}
    assert len(new['recovery_routes'])==4 and new['original_completion_configuration']==ORIGINAL
    assert new['external_completed_recoveries']==[dict(canonical_id='crossad_author__release_checkpoint__psm__seed2025',
        post_review_configuration='configs/reproducibility/fm_crossad_PSM_independent_metric_keywords.2026-10-12_v7.json')]
    assert new['metric_tolerance']==dict(rtol=1e-12,atol=1e-12)


def test_cold_handoff_config_pins_new_worker_and_exact_old_PID_time_and_command():
    config=original.read(HANDOFF)
    assert config['original_scope']==original.SCOPE
    assert config['required_original_jobs']==139 and config['required_original_reconstructed_ablation_jobs']==90
    for role,path in [('original',ORIGINAL),('recovery',RECOVERY)]:
        route=config['roles'][role];owner=route['previous_owner']
        assert route['controller_configuration']==path
        assert owner['pid'] and owner['create_time'] and '--config' in owner['command']
    sources={p['path']:p['sha256'] for p in config['source_receipts']}
    for path in ('scripts/flow_matching/run_crossad_post_fixed_controllers_v2.py',
            'scripts/flow_matching/replay_crossad_post_metric_keywords_v2.py'):
        assert sources[path]==original.digest(path)
