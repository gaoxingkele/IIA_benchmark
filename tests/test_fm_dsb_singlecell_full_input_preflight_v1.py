from scripts.flow_matching import preflight_dsb_singlecell_full_inputs_v1 as s


def test_every_existing_full_input_and_all12_leaveouts_have_full_source_budgets():
    c = s.registration()
    assert len(c['cases']) == 12 and len({v['dataset'] for v in c['cases']}) == 5
    assert (c['actual_cache_particles'], c['actual_cache_steps'], c['actual_cache_rows']) == (10000, 20, 200000)
    assert c['actual_training_batch'] == 512 and c['optimizer_updates'] == 0
    assert c['full_training_models_pending'] == 60 and c['full_updates_per_model_pending'] == 400040


def test_existing_author_environment_and_every_original_memory_guard_remain():
    c = s.registration()
    original = s.read(s.ROOT / s.INPUT)
    for key in ['settings', 'resource_lock', 'environment_lock', 'python', 'cases', 'train_val_test_split']:
        assert c[key] == original[key]
    assert c['settings']['minimum_commit_headroom_bytes'] == 32 * 2**30
    assert not c['historical_paper_equivalence_certified']
    assert not c['original_DSB_2021_environment_reproduced']
    assert not c['source_gamma_to_paper_sigma_mapping_certified']


def test_source_inclusive_iteration_budget_and_two_complete_networks_are_explicit():
    p = s.read(s.ROOT / s.MODEL)
    budget = p['paper_training_budget']
    assert budget['original_source_total_optimizer_updates'] == 20 * 2 * 10001
    assert budget['cache_particles'] == 10000 and budget['training_batch'] == 512
    assert p['network']['parameters_per_network'] == 14261 and p['network']['total_parameters'] == 28522
    assert not p['full_native_IPF_training_complete'] and not p['full_checkpoint_replay_complete']
