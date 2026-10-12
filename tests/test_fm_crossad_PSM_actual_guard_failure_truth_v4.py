from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.register_crossad_PSM_lifetime_completion_v1 import CONFIG,NATIVE


def test_actual_full_model_replay_is_failed_not_promoted_by_layer_diagnostics():
    config=full.read(CONFIG);native=full.read(NATIVE)
    diagnostic=full.read(config['diagnostic_terminal'])
    terminal=full.read(full.ROOT/native['state_root']/'receipt.json')
    assert diagnostic['passed'] and diagnostic['exit_code']==0
    assert terminal['exit_code']==1 and terminal['resource_guard_aborted']
    assert not terminal['full_native_model_replay_passed']
    assert terminal['peak_tree_rss_bytes']>native['settings']['maximum_worker_rss_bytes']
    assert native['settings']['maximum_worker_rss_bytes']==3*2**30
    assert native['settings']['maximum_worker_private_bytes']==4*2**30
    assert not (full.ROOT/native['artifact_directory']/'full_native_model_replay.json').exists()


def test_idle_CFM_handoff_preserves_actual_full400_training_and_every_terminal_route():
    from scripts.flow_matching.run_cfm_checkpointed_light_parent_v2 import lightweight_namespace
    light=lightweight_namespace()
    config=light['read'](full.ROOT/'configs/runtime/fm_cfm_checkpointed_parent_import_handoff.v2.json')
    producer=light['read'](full.ROOT/config['producer_configuration'])
    queue,job=light['verify_registration'](producer)
    assert len(queue['jobs'])==285 and job['epochs']==job['expected_optimizer_updates']==400
    assert job['seed']==1105
    proof=light['read'](full.ROOT/config['state_root']/'handoff.json')
    assert proof['no_native_child_present'] and proof['waiting_controller_only_stopped']
    assert not proof['original_native_child_stopped']
    replay=light['read'](full.ROOT/'configs/runtime/fm_cfm_completion_replay_queue.v2.json')
    route=next(r for r in replay['routes'] if r['original_canonical_id']==producer['canonical_job']['id'])
    assert route['checkpointed_training_terminal']==producer['state_root']+'/receipt.json'
