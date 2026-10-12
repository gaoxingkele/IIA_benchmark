"""Reuse Windows telemetry ctypes types without changing any resource formula."""
import ctypes
import sys

import psutil


class PerformanceInfo(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in
        ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
         'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
        (n, ctypes.c_ulong) for n in ('HandleCount', 'ProcessCount', 'ThreadCount')]


_performance_library = None
_performance_api = None


def windows_api():
    global _performance_library, _performance_api
    if _performance_api is None:
        _performance_library = ctypes.WinDLL('psapi', use_last_error=True)
        api = _performance_library.GetPerformanceInfo
        api.argtypes = [ctypes.POINTER(PerformanceInfo), ctypes.c_ulong]
        api.restype = ctypes.c_int
        _performance_api = api
    return _performance_api


def resource_snapshot():
    available = psutil.virtual_memory().available
    if sys.platform == 'win32':
        info = PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        if not windows_api()(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        commit = (info.CommitLimit - info.CommitTotal) * info.PageSize
    else:
        commit = available + psutil.swap_memory().free
    return dict(physical_available_bytes=available, commit_available_bytes=commit)
