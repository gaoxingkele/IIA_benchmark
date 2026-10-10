from types import SimpleNamespace

import pytest

from scripts.flow_matching.full_gpu_memory_slot_v2 import try_full_gpu_slot, full_gpu_memory_slot


def fake_api(memory_sequence, gpu_sequence):
    calls = []
    handle = SimpleNamespace(close=lambda: calls.append('closed'))
    memory = iter(memory_sequence)
    gpu = iter(gpu_sequence)
    api = dict(resource_snapshot=lambda: next(memory), gpu_snapshot=lambda settings: next(gpu),
        has_headroom=lambda m, s: m['physical_available_bytes'] >= s['minimum_available_memory_bytes']
        and m['commit_available_bytes'] >= s['minimum_commit_headroom_bytes'],
        exclusive_lock=lambda path: calls.append(str(path)) or handle)
    return api, calls, handle


def registration():
    return dict(resource_lock='gpu.lock', cpu_reservation_lock='occupied_cpu.lock',
        settings=dict(minimum_available_memory_bytes=32, minimum_commit_headroom_bytes=42, minimum_gpu_free_bytes=14))


def test_original_full_memory_gates_are_rechecked_after_gpu_lock():
    ample = dict(physical_available_bytes=40, commit_available_bytes=50)
    depleted = dict(physical_available_bytes=40, commit_available_bytes=41)
    api, calls, _ = fake_api([ample, depleted], [dict(gpu_free_bytes=24)]*2)
    handle, _ = try_full_gpu_slot(api, registration())
    assert handle is None and calls[-1] == 'closed'


@pytest.mark.parametrize('resource', ['physical', 'commit', 'gpu'])
def test_below_full_original_memory_gate_never_launches(resource):
    memory = dict(physical_available_bytes=40, commit_available_bytes=50)
    gpu = dict(gpu_free_bytes=24)
    if resource == 'physical': memory['physical_available_bytes'] = 31
    if resource == 'commit': memory['commit_available_bytes'] = 41
    if resource == 'gpu': gpu['gpu_free_bytes'] = 13
    api, calls, _ = fake_api([memory], [gpu])
    handle, _ = try_full_gpu_slot(api, registration())
    assert handle is None and not calls


def test_full_gpu_work_can_coexist_with_cpu_training_and_releases_on_failure():
    memory = dict(physical_available_bytes=40, commit_available_bytes=50)
    api, calls, _ = fake_api([memory]*2, [dict(gpu_free_bytes=24)]*2)
    with pytest.raises(RuntimeError, match='worker failure'):
        with full_gpu_memory_slot(api, registration(), lambda values: None):
            raise RuntimeError('worker failure')
    assert len(calls)==2 and calls[-1]=='closed'
    assert not any('occupied_cpu.lock' in call for call in calls)
