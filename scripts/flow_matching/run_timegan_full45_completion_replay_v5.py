"""Keep every native replay gate while removing its host-only psutil import.

The full original replay worker, 1035 checkpoints, 900 evaluators, all native
sampling iterations, bitwise equality and resource limits remain unchanged.
"""
import ast
import copy

from scripts.flow_matching import run_timegan_full45_completion_replay_v4 as original
from scripts.flow_matching.timegan_native_result_verifier_v2 import verify_result

BASE = original.original.original.base
PREVIOUS = 'configs/runtime/fm_timegan_full45_completion_replay.v5.json'
DISPATCH = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v3.json'
COMPLETION_RECEIPT = 'projects/flow_matching_research/results/2026-10-13/timegan_energy1104_actual_completion_v1/independent_completion_verification.json'


def completed_native_result(route, queue):
    if route['kind'] == 'original':
        return verify_result(queue, route['execution_job'])
    return original.original.completed_native_result(route, queue)


def expected_routes(previous):
    routes = copy.deepcopy(previous['routes'])
    for route in routes:
        if route['kind'] != 'original':
            continue
        identity = route['canonical_job']['id']
        route['native_completion_receipts'].append(dict(
            path='experiments/runs/fm_timegan_corrected_observer_original_v3/receipts/' + identity + '.json',
            successful_field='full_native_result_verified'))
        if identity == 'timegan_native__gru__energy__seed1104':
            route['native_completion_receipts'].append(dict(
                path=COMPLETION_RECEIPT, successful_field='full_native_result_verified'))
    return routes


def verify(config):
    namespace = original.runtime_namespace()
    queue, routes = namespace['verify'](config)
    previous = BASE.read(BASE.ROOT / PREVIOUS)
    if routes != expected_routes(previous):
        raise ValueError('All45 exact original routes remain; only verified successor receipts are added')
    for field in ('original_queue', 'settings', 'resource_lock', 'python', 'cpu_threads',
                  'required_checkpoints_each', 'required_evaluators_each', 'equality_gate',
                  'protocol_iterations_each', 'required_full_generated_windows',
                  'required_full_predictive_test_windows', 'required_full_discriminative_test_windows'):
        if config[field] != previous[field]:
            raise ValueError('Original full replay budgets and guards changed: ' + field)
    return queue, routes


def main_ast():
    tree = original.main_ast()
    changes = [0]

    class WorkerRoute(ast.NodeTransformer):
        def value(self, value):
            old = "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v4'"
            if isinstance(value, str) and old in value:
                changes[0] += 1
                return value.replace(old, "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v5'")
            return value

        def visit_Constant(self, node):
            node.value = self.value(node.value)
            return node

        def visit_Str(self, node):
            node.s = self.value(node.s)
            return node

    tree = WorkerRoute().visit(tree)
    if changes != [1]:
        raise ValueError('One original replay worker entry point required')
    return ast.fix_missing_locations(tree)


def runtime_namespace():
    namespace = original.runtime_namespace()
    namespace.update(completed_native_result=completed_native_result, verify=verify)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    runtime_namespace()['main']()
