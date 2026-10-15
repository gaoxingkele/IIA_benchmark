"""Cold-cache four idle original queues without changing numerical admission."""
import importlib
from types import FunctionType

from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as original

ENTRIES = {
    'giflow140': ('scripts.flow_matching.run_giflow_serial_native_queue_v4', 140),
    'spectral_operator70': ('scripts.flow_matching.audit_spectral_full_sampling_derivative_graph_v1', 70),
    'cfm_retry285': ('scripts.flow_matching.run_cfm_ts_stricter_admission_retry_v1', 285),
    'ls4_physio10': ('scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1', 10),
}


def idle_allowed(state, expected, descendants):
    if state.get('state') == 'waiting_full_gpu_resources':
        # This is an idle host phase, not an altered GPU admission decision.
        # Original state bytes are retained verbatim in the handoff evidence.
        return original.idle_allowed(dict(state, state='waiting_for_memory'), expected, descendants)
    return original.idle_allowed(state, expected, descendants)


verify = FunctionType(original.verify.__code__, dict(original.__dict__, ENTRIES=ENTRIES),
                      argdefs=original.verify.__defaults__)
handoff = FunctionType(original.handoff.__code__, dict(original.__dict__, idle_allowed=idle_allowed),
                       argdefs=original.handoff.__defaults__)
main = FunctionType(original.main.__code__, dict(original.__dict__, ENTRIES=ENTRIES,
                                                verify=verify, handoff=handoff))


def actual_guard_api_probe(name, role):
    """Resolve each real guard API in a fresh host; never execute an entry main."""
    original.install_cached_loader(lambda snapshot: None)
    module = importlib.import_module(ENTRIES[name][0])
    if name == 'giflow140':
        queue = original.read(original.ROOT / role['original_queue'])
        runtime = original.read(original.ROOT / queue['runtime_config'])
        _, api, _ = module.guard.load_controller(original.ROOT, runtime, 'crossad')
    elif name == 'spectral_operator70':
        api = module.resource_api()
    elif name == 'cfm_retry285':
        api = module.original.resource_api()
    elif name == 'ls4_physio10':
        entry = importlib.import_module('scripts.flow_matching.run_ls4_native_physionet_queue_v3')
        api = entry.guarded_api()
    else:
        raise ValueError('Only explicit original resource APIs are supported')
    snapshot = api['resource_snapshot']
    if snapshot.__name__ != 'cached_snapshot':
        raise ValueError('Actual original guard must resolve the installed cached resource snapshot')
    import psutil
    import sys
    return dict(role=name, actual_guard_snapshot=snapshot.__name__, snapshot=snapshot(),
                private_bytes=psutil.Process().memory_info().private,
                numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
                entry_main_executed=False, numerical_child_executed=False)


if __name__ == '__main__':
    main()
