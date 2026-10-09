"""Repair the recovery verifier's return contract, preserving frozen worker code."""
from types import FunctionType

from scripts.flow_matching import native_exact_recovery as original


def verified_manifest(queue, root=original.ROOT):
    registered = original.verify(queue, root)
    if queue['recovery_track'] != 'grasp':
        raise ValueError('This repair is registered only for GRASP')
    worker, _ = original.components('grasp')
    return worker.verify(registered, root)


def worker_main():
    bound = original.worker_main('grasp')
    bound.__globals__['verify'] = verified_manifest
    return bound


def controller_main():
    bound = original.controller_main('grasp')
    source = 'scripts.flow_matching.run_native_exact_recovery_job'
    replacement = 'scripts.flow_matching.run_native_grasp_recovery_v2'
    if bound.__code__.co_consts.count(source) != 1:
        raise ValueError('Frozen recovery dispatch changed')
    constants = tuple(replacement if c == source else c for c in bound.__code__.co_consts)
    bound.__globals__['verify'] = verified_manifest
    return FunctionType(bound.__code__.replace(co_consts=constants), bound.__globals__,
                        bound.__name__, bound.__defaults__, bound.__closure__)
