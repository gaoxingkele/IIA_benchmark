"""Serialize heavy jobs and guard Windows system commit without reducing budgets."""
import ctypes
from contextlib import contextmanager
import sys
import time
import psutil
from scripts.flow_matching.run_tsad_queue import exclusive_lock


def resource_snapshot():
    available = psutil.virtual_memory().available
    if sys.platform == 'win32':
        class PerformanceInfo(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in
                ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
                 'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
                (n, ctypes.c_ulong) for n in ('HandleCount', 'ProcessCount', 'ThreadCount')]
        info = PerformanceInfo(); info.cb = ctypes.sizeof(info)
        api = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
        api.argtypes = [ctypes.POINTER(PerformanceInfo), ctypes.c_ulong]; api.restype = ctypes.c_int
        if not api(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        commit = (info.CommitLimit-info.CommitTotal)*info.PageSize
    else:
        commit = available + psutil.swap_memory().free
    return {'physical_available_bytes': available, 'commit_available_bytes': commit}


def has_headroom(snapshot, settings):
    return snapshot['physical_available_bytes'] >= settings['minimum_available_memory_bytes'] and \
        snapshot['commit_available_bytes'] >= settings['minimum_commit_headroom_bytes']


@contextmanager
def waited_lock(path, waiting=None, poll_seconds=10):
    handle = exclusive_lock(path)
    while handle is None:
        if waiting:
            waiting({'state': 'waiting_for_heavy_lock'})
        time.sleep(poll_seconds)
        handle = exclusive_lock(path)
    try:
        yield handle
    finally:
        handle.close()


def wait_for_headroom(settings, update=None, poll_seconds=10):
    state = resource_snapshot()
    while not has_headroom(state, settings):
        if update:
            update(dict(state, state='waiting_for_memory'))
        time.sleep(poll_seconds); state = resource_snapshot()
    return state


def terminate_tree(process):
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def guarded_wait(process, settings, update=None, poll_seconds=1):
    peaks = {'peak_tree_rss_bytes': 0, 'peak_tree_private_bytes': 0,
             'minimum_commit_available_bytes': None, 'resource_guard_aborted': False}
    while process.poll() is None:
        memory = resource_snapshot()
        current = memory['commit_available_bytes']
        peaks['minimum_commit_available_bytes'] = min(current, peaks['minimum_commit_available_bytes'] or current)
        try:
            parent = psutil.Process(process.pid)
            entries = [parent]+parent.children(recursive=True)
            values = [p.memory_info() for p in entries]
            peaks['peak_tree_rss_bytes'] = max(peaks['peak_tree_rss_bytes'], sum(v.rss for v in values))
            peaks['peak_tree_private_bytes'] = max(peaks['peak_tree_private_bytes'], sum(getattr(v, 'private', v.vms) for v in values))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if current < settings['emergency_commit_headroom_bytes']:
            peaks.update(resource_guard_aborted=True, abort_resource_snapshot=memory)
            terminate_tree(process)
            if update:
                update(dict(peaks, state='resource_guard_aborted'))
            break
        time.sleep(poll_seconds)
    return process.wait(), peaks
