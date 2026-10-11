"""Verify dispatch isolation cannot alter the original full scientific budget."""
import copy
import pytest
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.run_cfm_ts_parallel_native_slot_v1 import verify_runtime, verify_controller


def test_original_recovery_job_and_every_frozen_input_are_unchanged():
    runtime = read(ROOT / 'configs/runtime/fm_cfm_ts_low_memory_controller.v9.json')
    old = read(ROOT / runtime['predecessor_runtime'])
    queue, job = verify_runtime(runtime)
    assert runtime['controllers'] == old['controllers']
    assert runtime['source_receipts'][:len(old['source_receipts'])] == old['source_receipts']
    assert len(queue['jobs']) == 285
    assert job['id'].endswith('seed1104__exact_recovery_v3')
    assert queue['resource_policy']['cpu_threads'] == 1
    assert runtime['parallel_admission_policy']['emergency_commit_headroom_bytes'] >= queue['resource_policy']['emergency_commit_headroom_bytes']


def test_no_second_slot_or_weaker_commit_protection_can_enter_parallel_lane():
    runtime = read(ROOT / 'configs/runtime/fm_cfm_ts_low_memory_controller.v9.json')
    bad = copy.deepcopy(runtime)
    bad['controllers']['recovery']['selected_job_ids'] *= 2
    with pytest.raises(ValueError, match='Only'):
        verify_runtime(bad)
    bad = copy.deepcopy(runtime)
    bad['parallel_admission_policy']['emergency_commit_headroom_bytes'] = 0
    with pytest.raises(ValueError, match='weaken'):
        verify_runtime(bad)


def test_original_full_queue_cannot_enter_independent_recovery_lane():
    for arguments in [[], ['--controller', 'original']]:
        with pytest.raises(ValueError, match='explicit recovery'):
            verify_controller(arguments)
    verify_controller(['--controller', 'recovery'])


def test_worker_caps_are_positive_and_cannot_expand():
    runtime = read(ROOT / 'configs/runtime/fm_cfm_ts_low_memory_controller.v9.json')
    for key, value in [('maximum_worker_private_bytes', 0), ('maximum_worker_rss_bytes', 9*1024**3)]:
        bad = copy.deepcopy(runtime)
        bad['parallel_admission_policy'][key] = value
        with pytest.raises(ValueError, match='capped'):
            verify_runtime(bad)
