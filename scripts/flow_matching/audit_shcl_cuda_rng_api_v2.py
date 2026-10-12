"""Observe the exact failed native API stages; retain all computations and caps."""
import ast
import ctypes
from ctypes import wintypes
import inspect
import os

from scripts.flow_matching import audit_shcl_cuda_rng_api_v1 as original


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage',
            'PeakPagefileUsage', 'PrivateUsage')]


def observer(config):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    rows = []
    def observe(phase, result=None):
        info = ProcessMemory()
        info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        rows.append(dict(phase=phase, private_bytes=info.PrivateUsage, rss_bytes=info.WorkingSetSize,
            completed_native_API_result=result))
        original.write(original.job_api.replay.original.native.ROOT / config['state_root'] / 'native_operation_memory.json',
            dict(worker_pid=os.getpid(), observations=rows, original_guard_settings=config['settings'],
                synthetic_operator_only_not_benchmark_performance=True, canonical_training_credit=0))
    return observe


def worker_ast():
    tree = ast.parse(inspect.getsource(original.worker))
    function = tree.body[0]
    changes = 0
    before = {
        ast.dump(ast.parse('torch.cuda.get_rng_state_all()').body[0]): 'before_native_CUDA_RNG_initialization',
        ast.dump(ast.parse('result = job_api.replay.restore_and_verify_draws(checkpoint)').body[0]): 'before_complete_native_rng_restore',
    }
    after = {
        ast.dump(ast.parse('import torch').body[0]): 'after_native_torch_import',
        ast.dump(ast.parse('torch.set_num_threads(2)').body[0]): 'after_native_thread_configuration',
        ast.dump(ast.parse('torch.cuda.get_rng_state_all()').body[0]): 'after_native_CUDA_RNG_initialization',
    }
    body = []
    for node in function.body:
        match = ast.dump(node)
        if match in before:
            body += ast.parse('observe(' + repr(before[match]) + ')').body
            changes += 1
        body.append(node)
        if match in after:
            body += ast.parse('observe(' + repr(after[match]) + ')').body
            changes += 1
        if match == ast.dump(ast.parse('assert observation.backend_flags() == original_flags').body[0]):
            body += ast.parse("observe('after_full_rng_and_backend_assertions', result)").body
            changes += 1
    if changes != 7:
        raise ValueError('Exactly seven observation statements surround the same native operations')
    function.body = body
    return ast.fix_missing_locations(tree)


def worker(config, path):
    namespace = dict(original.__dict__, observe=observer(config))
    exec(compile(worker_ast(), __file__, 'exec'), namespace)
    namespace['worker'](config, path)


def main():
    tree = ast.parse(inspect.getsource(original.main))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'scripts.flow_matching.audit_shcl_cuda_rng_api_v1':
            node.value = 'scripts.flow_matching.audit_shcl_cuda_rng_api_v2'
            changes += 1
    if changes != 1:
        raise ValueError('Only the fresh observed native worker entry changes')
    namespace = dict(original.__dict__, worker=worker)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
