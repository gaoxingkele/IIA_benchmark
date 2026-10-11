import ast
from pathlib import Path

import pytest

from scripts.flow_matching.run_cfm_ts_stricter_admission_retry_v1 import controller_ast


def test_actual_controller_preserves_every_statement_except_admission_argument():
    source = Path('scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py').read_text(encoding='utf-8')
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    adapted = controller_ast(source)
    compile(adapted, '<cfm-stricter-admission>', 'exec')
    replacements = []
    class Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_retry_admission_settings':
                assert node.args[0].id == 'runtime'
                assert ast.unparse(node.args[1]) == "queue['resource_policy']"
                replacements.append(node)
                return node.args[1]
            return self.generic_visit(node)
    assert ast.dump(Restore().visit(adapted.body[0])) == ast.dump(original)
    assert len(replacements) == 1


@pytest.mark.parametrize('source', [
    'def main(): pass',
    "def main():\n api['resource_slot'](lock, wrong, update)",
    "def main():\n api['resource_slot'](lock, queue['resource_policy'])",
    "def main():\n api['resource_slot'](lock, queue['resource_policy'], update)\n api['resource_slot'](lock, queue['resource_policy'], update)"])
def test_missing_or_changed_resource_boundary_rejected(source):
    with pytest.raises(ValueError):
        controller_ast(source)
