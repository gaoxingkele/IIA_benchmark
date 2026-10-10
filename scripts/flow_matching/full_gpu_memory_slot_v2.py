"""Admit one full GPU pipeline using unchanged RAM/commit/VRAM safeguards."""
from contextlib import contextmanager
import subprocess
import time

from scripts.flow_matching.run_light_controller import ROOT


def try_full_gpu_slot(api, registration):
    settings = registration['settings']
    memory = api['resource_snapshot']()
    gpu = api['gpu_snapshot'](settings)
    if not api['has_headroom'](memory, settings) or gpu['gpu_free_bytes'] < settings['minimum_gpu_free_bytes']:
        return None, dict(memory, **gpu)
    handle = api['exclusive_lock'](ROOT / registration['resource_lock'])
    if handle is None:
        return None, dict(memory, **gpu)
    try:
        memory = api['resource_snapshot']()
        gpu = api['gpu_snapshot'](settings)
        if api['has_headroom'](memory, settings) and gpu['gpu_free_bytes'] >= settings['minimum_gpu_free_bytes']:
            return handle, dict(memory, **gpu)
    except BaseException:
        handle.close()
        raise
    handle.close()
    return None, dict(memory, **gpu)


@contextmanager
def full_gpu_memory_slot(api, registration, update):
    # GPU actors remain serialized by the original GPU mutex. The shared CPU
    # mutex serializes CPU training, but is not a GPU memory reservation. Check
    # the same full physical/commit/VRAM gates again after acquiring the GPU
    # mutex, and retain the original guarded_gpu_wait emergency protections.
    while True:
        try:
            handle, observation = try_full_gpu_slot(api, registration)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as error:
            update(dict(state='waiting_gpu_observation_retry', gpu_observation_error=type(error).__name__))
            time.sleep(10)
            continue
        if handle is not None:
            break
        update(dict(state='waiting_full_gpu_resources', shared_lock_owned=False, **observation))
        time.sleep(10)
    try:
        yield
    finally:
        handle.close()
