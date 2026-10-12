"""Acquire both original CPU replay and shared GPU mutexes before any CUDA work."""
from contextlib import contextmanager
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.run_shcl_native_script_queue_v1 import gpu_snapshot


def verify_gpu_mutex(config):
    reference = read(ROOT / config['GPU_mutex_reference_configuration'])
    if (config['GPU_resource_lock'] != reference['resource_lock']
            or config['GPU_resource_lock'] == config['resource_lock']
            or config['gpu_settings']['gpu_uuid'] != reference['settings']['gpu_uuid']):
        raise ValueError('Original shared training GPU mutex and distinct CPU replay lane required')


def try_combined_slot(api, config, fixture=False):
    held = []
    try:
        paths = [] if fixture else [ROOT / config['GPU_resource_lock']]
        paths.append(ROOT / config['resource_lock'])
        for path in paths:
            handle = api['exclusive_lock'](path)
            if handle is None:
                return None, dict(state='waiting_for_shared_GPU_or_CPU_mutex', blocked_lock=str(path))
            held.append(handle)
        snapshot = api['resource_snapshot']()
        gpu = {} if fixture else api['gpu_snapshot'](config['gpu_settings'])
        if not api['has_headroom'](snapshot, config['settings']) or (gpu and gpu['gpu_free_bytes'] < config['gpu_settings']['minimum_gpu_free_bytes']):
            return None, dict(snapshot, **gpu, state='waiting_for_original_readonly_capacity')
        confirmed = api['resource_snapshot']()
        gpu = {} if fixture else api['gpu_snapshot'](config['gpu_settings'])
        if not api['has_headroom'](confirmed, config['settings']) or (gpu and gpu['gpu_free_bytes'] < config['gpu_settings']['minimum_gpu_free_bytes']):
            return None, dict(confirmed, **gpu, state='waiting_for_original_readonly_capacity')
        result = tuple(held)
        held = []
        return result, dict(confirmed, **gpu, state='original_shared_GPU_and_CPU_lanes_reserved')
    finally:
        for handle in reversed(held):
            handle.close()


@contextmanager
def combined_slot(api, config, update, fixture=False):
    verify_gpu_mutex(config)
    wrapped = dict(api, gpu_snapshot=gpu_snapshot)
    while True:
        handles, observation = try_combined_slot(wrapped, config, fixture)
        if handles is not None:
            break
        update(dict(observation, shared_slot_owned=False))
        time.sleep(10)
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.close()
