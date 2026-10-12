"""Use the exact verified full-stage recovery with corrected target observation.

Original guards, completed-stage restoration, joint and all20 evaluator budgets
are preserved. Only the worker module dispatch and one observation are adapted.
"""
import ast
import inspect

from scripts.flow_matching import run_timegan_stage_resume_v2 as original
from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_resume_compiler


def main_ast():
    tree = ast.parse(inspect.getsource(original.main))
    changes = [0]

    class WorkerRoute(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and "runpy.run_module('scripts.flow_matching.run_timegan_stage_resume_v2'" in node.value:
                changes[0] += 1
                node.value = node.value.replace('scripts.flow_matching.run_timegan_stage_resume_v2',
                    'scripts.flow_matching.run_timegan_corrected_stage_recovery_v3')
            return node

        def visit_Str(self, node):
            if "runpy.run_module('scripts.flow_matching.run_timegan_stage_resume_v2'" in node.s:
                changes[0] += 1
                node.s = node.s.replace('scripts.flow_matching.run_timegan_stage_resume_v2',
                    'scripts.flow_matching.run_timegan_corrected_stage_recovery_v3')
            return node

    tree = WorkerRoute().visit(tree)
    if changes != [1]:
        raise ValueError('Exactly one native corrected stage worker module dispatch required')
    return ast.fix_missing_locations(tree)


def main():
    namespace = dict(original.__dict__, compile_observed_resume=corrected_resume_compiler())
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
