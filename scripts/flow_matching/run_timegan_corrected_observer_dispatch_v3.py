"""Separate host monitoring from original TF1 verification; retain all45 jobs.

Immutable successor to v2. Only the idle original controller is handed off.
Existing Stock/Energy recovery controllers and scientific source stay frozen.
"""
import ast
import inspect

from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as original
from scripts.flow_matching.timegan_native_result_verifier_v2 import observer_worker

PREVIOUS = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v2.json'


def verify_runtime(config):
    queue = original.verify_runtime(config)
    prior = original.read(original.ROOT / PREVIOUS)
    if (config['original_queue'] != prior['original_queue']
            or config['resource_lock'] != prior['resource_lock']
            or config['cpu_threads'] != prior['cpu_threads']
            or config['poll_seconds'] != prior['poll_seconds']
            or config['roles']['stock'] != prior['roles']['stock']
            or config['roles']['energy'] != prior['roles']['energy']
            or config['roles']['original']['settings'] != prior['roles']['original']['settings']
            or config['roles']['original']['recovery_configuration'] is not None
            or config['native_result_verification'] != 'exact_original_function_without_host_monitor_import'):
        raise ValueError('Full original queue, algorithms, other controllers and resource guards stay frozen')
    return queue


def main_ast():
    tree = ast.parse(inspect.getsource(original.main))
    changes = [0]

    class WorkerRoute(ast.NodeTransformer):
        def value(self, value):
            old = "runpy.run_module('scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2'"
            if isinstance(value, str) and old in value:
                changes[0] += 1
                return value.replace(old, "runpy.run_module('scripts.flow_matching.run_timegan_corrected_observer_dispatch_v3'")
            return value

        def visit_Constant(self, node):
            node.value = self.value(node.value)
            return node

        def visit_Str(self, node):
            node.s = self.value(node.s)
            return node

    tree = WorkerRoute().visit(tree)
    if changes != [1]:
        raise ValueError('Only one native worker entry point changes')
    return ast.fix_missing_locations(tree)


def runtime_namespace():
    namespace = dict(original.__dict__, verify_runtime=verify_runtime, worker=observer_worker())
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace


def main():
    import sys
    if '--role' not in sys.argv or sys.argv[sys.argv.index('--role') + 1] != 'original':
        raise ValueError('Only the original controller needs this immutable successor')
    runtime_namespace()['main']()


if __name__ == '__main__':
    main()
