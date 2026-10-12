"""Immutable full45 replay route extension for a verified posthoc recovery.

The v1 numerical replay worker, bitwise gates, loops and caps are retained.
Only route dispatch and the verified idle controller handoff are extended.
"""
import ast
import copy
import inspect
from types import FunctionType

from scripts.flow_matching import run_timegan_full45_completion_replay_v1 as base
from scripts.flow_matching.timegan_posthoc_recovery_v1 import binding, verify_recovery_result


def check_routes(queue, config):
    normalized = copy.deepcopy(config)
    for route in normalized['routes']:
        if route['kind'] == 'posthoc_recovery':
            recovery = base.read(base.ROOT / route['recovery_configuration'])
            verified_queue, model, job, checkpoints, evaluators = binding(recovery)
            if verified_queue != queue or route['canonical_job'] != job:
                raise ValueError('Exact original full45 posthoc recovery binding required')
            route['kind'] = 'stage_resume'
    # Every existing complete scope, canonical identity, shape and guard check stays.
    base.check_routes(queue, normalized)
    return config['routes']


def completed_native_result(route, queue):
    if route['kind'] == 'posthoc_recovery':
        return verify_recovery_result(base.read(base.ROOT / route['recovery_configuration']))
    return base.completed_native_result(route, queue)


def handoff_phase_allowed(role, state, output_exists, descendants):
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child
    return (state.get('active_pid') is None and state.get('active_job') is None
        and state.get('state') == 'waiting_actual_complete_original_full45_native_results'
        and not state.get('verified_original_ids') and not state.get('failed')
        and all(allowed_console_child(p['name'], p['exe']) for p in descendants))


def verified_replay_handoff(config, state_root, update):
    from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as dispatch
    # An isolated function namespace retains all PID/ctime/command rechecks and
    # suspend/recheck/finally-resume safeguards. No live script is mutated.
    namespace = dict(dispatch.__dict__, handoff_phase_allowed=handoff_phase_allowed)
    handoff = FunctionType(dispatch.safe_handoff.__code__, namespace)
    route = dict(previous_owner=config['previous_replay_owner'],
        previous_state_path=config['previous_replay_state_path'],
        runtime_configuration=config['runtime_configuration'])
    handoff('original', route, state_root, update, config['poll_seconds'])


def main_ast():
    tree = ast.parse(inspect.getsource(base.main))
    changes = [0]

    class WorkerRoute(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v1'" in node.value:
                changes[0] += 1
                node.value = node.value.replace('scripts.flow_matching.run_timegan_full45_completion_replay_v1',
                    'scripts.flow_matching.run_timegan_full45_completion_replay_v2')
            return node

        # Python 3.7 represents the bootstrap literal as ast.Str.
        def visit_Str(self, node):
            if "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v1'" in node.s:
                changes[0] += 1
                node.s = node.s.replace('scripts.flow_matching.run_timegan_full45_completion_replay_v1',
                    'scripts.flow_matching.run_timegan_full45_completion_replay_v2')
            return node

    tree = WorkerRoute().visit(tree)
    tries = [n for n in tree.body[0].body if isinstance(n, ast.Try)
             and n.body and isinstance(n.body[0], ast.While)]
    if changes != [1] or len(tries) != 1:
        raise ValueError('One native worker module route and one idle controller handoff required')
    tries[0].body.insert(0, ast.parse('_verified_replay_handoff(config, base, update)').body[0])
    return ast.fix_missing_locations(tree)


def runtime_namespace():
    namespace = dict(base.__dict__)
    # Rebind all v1 functions to an isolated namespace, preserving their code.
    for name, value in base.__dict__.items():
        if isinstance(value, FunctionType) and value.__module__ == base.__name__:
            replacement = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            replacement.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = replacement
    namespace.update(check_routes=check_routes, completed_native_result=completed_native_result,
                     _verified_replay_handoff=verified_replay_handoff)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace


def main():
    runtime_namespace()['main']()


if __name__ == '__main__':
    main()
