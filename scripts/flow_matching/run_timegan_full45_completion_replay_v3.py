"""Preserve all45 native replays while routing Sine's full checkpointed recovery.

Only an idle controller with zero verified/failed cases and no native child is
handed off. Every original numerical replay function and gate is retained.
"""
import ast
from types import FunctionType

from scripts.flow_matching import run_timegan_full45_completion_replay_v2 as original
from scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4 import cached_resource_api


def handoff_phase_allowed(role, state, output_exists, descendants):
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child
    return (state.get('active_pid') is None and state.get('active_job') is None
        and state.get('required_original_models') == 45 and state.get('pending_original_models') == 45
        and not state.get('verified_original_ids') and not state.get('failed')
        and state.get('state') in ('waiting_actual_complete_original_full45_native_results',
                                  'waiting_for_memory', 'waiting_for_heavy_lock')
        and all(allowed_console_child(p['name'], p['exe']) for p in descendants))


def verified_replay_handoff(config, state_root, update):
    from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as dispatch
    namespace = dict(dispatch.__dict__, handoff_phase_allowed=handoff_phase_allowed)
    handoff = FunctionType(dispatch.safe_handoff.__code__, namespace)
    route = dict(previous_owner=config['previous_replay_owner'],
        previous_state_path=config['previous_replay_state_path'],
        runtime_configuration=config['runtime_configuration'])
    handoff('original', route, state_root, update, config['poll_seconds'])


def completed_native_result(route, queue):
    if route['kind'] == 'stage_resume':
        config = original.base.read(original.base.ROOT / route['recovery_configuration'])
        if 'joint_checkpoint_interval' in config:
            from scripts.flow_matching.timegan_joint_checkpoint_observer_v1 import verify_checkpointed_recovery_result
            return verify_checkpointed_recovery_result(config)
    return original.completed_native_result(route, queue)


def main_ast():
    tree = original.main_ast()
    changes = dict(route=0, telemetry=0)
    class Binding(ast.NodeTransformer):
        def value(self, value):
            if isinstance(value, str) and "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v2'" in value:
                changes['route'] += 1
                return value.replace('scripts.flow_matching.run_timegan_full45_completion_replay_v2',
                                     'scripts.flow_matching.run_timegan_full45_completion_replay_v3')
            return value
        def visit_Constant(self, node):
            node.value = self.value(node.value)
            return node
        def visit_Str(self, node):
            node.s = self.value(node.s)
            return node
        def visit_ImportFrom(self, node):
            if node.module == 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
                if len(node.names) != 1 or node.names[0].name != 'resource_api':
                    raise ValueError('The host resource API alone may be rebound')
                changes['telemetry'] += 1
                return ast.Assign(targets=[ast.Name(id='resource_api', ctx=ast.Store())],
                                  value=ast.Name(id='_cached_resource_api', ctx=ast.Load()))
            return node
    tree = Binding().visit(tree)
    if changes != dict(route=1, telemetry=1):
        raise ValueError('Exactly one full45 worker route and host telemetry binding required')
    return ast.fix_missing_locations(tree)


def runtime_namespace():
    namespace = original.runtime_namespace()
    namespace.update(_verified_replay_handoff=verified_replay_handoff,
                     completed_native_result=completed_native_result,
                     _cached_resource_api=cached_resource_api)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    runtime_namespace()['main']()
