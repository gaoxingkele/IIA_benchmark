"""Retain all45 native jobs with stronger admission and one zero-stage retry."""
import ast
import inspect
from pathlib import Path
import sys

from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v3 as original
from scripts.flow_matching.timegan_native_result_verifier_v2 import observer_worker_ast
from scripts.flow_matching.gan_zero_checkpoint_retry_v1 import (
    execution_job, verify_zero_checkpoint_incident, stronger_slot_ast, stronger_admission)

PREVIOUS = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v3.json'


def verify_runtime(config):
    queue = original.verify_runtime(config)
    previous = original.original.read(original.original.ROOT / PREVIOUS)
    if config['roles']['original']['settings'] != previous['roles']['original']['settings']:
        raise ValueError('All original emergency caps and native settings stay frozen')
    stronger_admission(config, config['roles']['original']['settings'])
    incident = config['zero_checkpoint_retry']
    native = Path(sys.executable).resolve() == Path(queue['python']).resolve()
    verify_zero_checkpoint_incident(incident, check_live=not native)
    canonical = next(job for job in queue['jobs'] if job['id'] == incident['canonical_job']['id'])
    if canonical != incident['canonical_job']:
        raise ValueError('All45 original job identities remain unchanged')
    execution_job(config, canonical)
    return queue


def execution_jobs(config, jobs):
    return [execution_job(config, job) for job in jobs]


def worker_ast():
    tree = observer_worker_ast()
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'job' and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == 'next']
    if len(assignments) != 1:
        raise ValueError('Only the original canonical job output binding changes')
    assignments[0].value = ast.Call(func=ast.Name(id='_execution_job', ctx=ast.Load()),
        args=[ast.Name(id='config', ctx=ast.Load()), assignments[0].value], keywords=[])
    return ast.fix_missing_locations(tree)


def worker():
    from scripts.flow_matching import timegan_predictive_observer_v2 as observer
    namespace = dict(observer.__dict__, _execution_job=execution_job, verify_runtime=verify_runtime)
    exec(compile(worker_ast(), __file__, 'exec'), namespace)
    return namespace['worker']


def main_ast():
    tree = original.main_ast()
    changes = dict(route=0, jobs=0)

    class Binding(ast.NodeTransformer):
        def value(self, value):
            old = "runpy.run_module('scripts.flow_matching.run_timegan_corrected_observer_dispatch_v3'"
            if isinstance(value, str) and old in value:
                changes['route'] += 1
                return value.replace(old, "runpy.run_module('scripts.flow_matching.run_timegan_corrected_observer_dispatch_v4'")
            return value

        def visit_Constant(self, node):
            node.value = self.value(node.value)
            return node

        def visit_Str(self, node):
            node.s = self.value(node.s)
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'jobs':
                changes['jobs'] += 1
                node.value = ast.Call(func=ast.Name(id='_execution_jobs', ctx=ast.Load()),
                    args=[ast.Name(id='config', ctx=ast.Load()), node.value], keywords=[])
            return self.generic_visit(node)

    tree = Binding().visit(tree)
    if changes != dict(route=1, jobs=1):
        raise ValueError('One native entry route and one full45 output mapping required')
    return stronger_slot_ast(tree, 'config', "route['settings']")


def runtime_namespace():
    namespace = dict(original.original.__dict__, verify_runtime=verify_runtime,
        worker=worker(), _execution_jobs=execution_jobs, _stronger_admission=stronger_admission)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace


def main():
    import sys
    if '--role' not in sys.argv or sys.argv[sys.argv.index('--role') + 1] != 'original':
        raise ValueError('Existing independent recovery controllers stay unchanged')
    runtime_namespace()['main']()


if __name__ == '__main__':
    main()
