"""Same original Sine1103 full budget, with complete joint-state observations."""
import ast
import inspect

from scripts.flow_matching import run_timegan_stage_resume_v2 as original
from scripts.flow_matching.timegan_joint_checkpoint_observer_v1 import (
    checkpoint_resume_compiler, verify_checkpointed_recovery_result)


def cached_resource_api():
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api = resource_api()
    api['resource_snapshot'] = resource_snapshot
    return api


def main_ast():
    tree = ast.parse(inspect.getsource(original.main))
    changes = dict(route=0, telemetry=0)
    class Binding(ast.NodeTransformer):
        def value(self, value):
            if isinstance(value, str) and "runpy.run_module('scripts.flow_matching.run_timegan_stage_resume_v2'" in value:
                changes['route'] += 1
                return value.replace('scripts.flow_matching.run_timegan_stage_resume_v2',
                                     'scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4')
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
                    raise ValueError('Only the host telemetry binding may change')
                changes['telemetry'] += 1
                return ast.Assign(targets=[ast.Name(id='resource_api', ctx=ast.Store())],
                                  value=ast.Name(id='_cached_resource_api', ctx=ast.Load()))
            return node
    tree = Binding().visit(tree)
    if changes != dict(route=1, telemetry=1):
        raise ValueError('The original complete worker route and cached host telemetry are required')
    return ast.fix_missing_locations(tree)


def main():
    namespace = dict(original.__dict__, compile_observed_resume=checkpoint_resume_compiler(),
        verify_recovery_result=verify_checkpointed_recovery_result, _cached_resource_api=cached_resource_api)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
