from scripts.flow_matching import combined_readonly_gpu_slot_v1 as slot


class Handle:
    def __init__(self): self.closed = False
    def close(self): self.closed = True


def config():
    return dict(GPU_resource_lock='gpu.lock', resource_lock='cpu.lock', settings=dict(minimum_commit_headroom_bytes=10),
        gpu_settings=dict(minimum_gpu_free_bytes=20))


def test_owned_GPU_mutex_prevents_even_initial_CUDA_observation_or_CPU_lane_acquisition():
    calls = []
    def lock(path): calls.append(str(path)); return None
    def unexpected(*args): raise AssertionError('No resource/CUDA work before GPU mutex')
    api = dict(exclusive_lock=lock, resource_snapshot=unexpected, gpu_snapshot=unexpected, has_headroom=unexpected)
    handles, state = slot.try_combined_slot(api, config())
    assert handles is None and len(calls) == 1 and calls[0].endswith('gpu.lock')


def test_busy_CPU_lane_releases_GPU_lane_and_never_retains_partial_reservation():
    held = Handle()
    api = dict(exclusive_lock=lambda path: held if str(path).endswith('gpu.lock') else None)
    handles, state = slot.try_combined_slot(api, config())
    assert handles is None and held.closed


def test_commit_or_GPU_capacity_loss_releases_both_original_lanes():
    for cause in ('commit', 'GPU'):
        held = []
        def lock(path):
            value = Handle(); held.append(value); return value
        api = dict(exclusive_lock=lock, resource_snapshot=lambda: {},
            has_headroom=lambda values, policy: cause != 'commit', gpu_snapshot=lambda policy: dict(gpu_free_bytes=0 if cause == 'GPU' else 20))
        handles, state = slot.try_combined_slot(api, config())
        assert handles is None and len(held) == 2 and all(value.closed for value in held)


def test_two_original_lanes_remain_owned_for_the_complete_worker_lifetime():
    held = []
    def lock(path):
        value = Handle(); held.append(value); return value
    api = dict(exclusive_lock=lock, resource_snapshot=lambda: {}, has_headroom=lambda values, policy: True,
        gpu_snapshot=lambda policy: dict(gpu_free_bytes=20))
    handles, state = slot.try_combined_slot(api, config())
    assert tuple(held) == handles and not any(value.closed for value in held)
    for value in reversed(handles): value.close()
    assert all(value.closed for value in held)
