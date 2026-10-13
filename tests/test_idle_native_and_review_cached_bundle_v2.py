from scripts.flow_matching import run_idle_native_and_review_cached_bundle_v2 as current


def test_reuse_existing_cold_handoff_without_altering_worker_or_guard_code():
    values=current.namespace()
    assert current.entry_main().__code__ is current.previous.main.__code__
    assert values['verify'].__code__ is current.previous.verify.__code__
    assert values['idle_allowed'] is current.previous.idle_allowed
    assert values['install_cached_loader'] is current.previous.install_cached_loader


def test_only_three_explicit_host_modules_are_added_without_mutating_live_v1():
    before=set(current.previous.ALLOWED)
    values=current.namespace()
    assert values['ALLOWED']-before==current.EXTRA_HOSTS
    assert current.previous.ALLOWED==before
    assert 'scripts.flow_matching.run_gtgan_energy_native_job_v1' not in values['ALLOWED']
    assert 'scripts.flow_matching.probe_tccm_full_image_execution_recovery_v1' not in values['ALLOWED']
