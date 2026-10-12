import ast
from pathlib import Path
import sys
import subprocess

import pytest

from scripts.flow_matching.rcgan_seed1105_resume_binding_v1 import MODULES, recovery_ast
from scripts.flow_matching.run_rcgan_seed1105_recovery_dispatch_v1 import DISPATCH_MODULES, dispatch_ast


@pytest.mark.parametrize('name', [
    'audit_rcgan_random_order_chain_v1.py', 'audit_rcgan_checkpoint_resume_state_v1.py',
    'rcgan_native_training_resume_runtime_v1.py', 'run_rcgan_native_state_recovery_v1.py'])
def test_all_scientific_statements_and_original_budgets_preserved(name):
    source = Path('scripts/flow_matching', name).read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)]
    actual = recovery_ast(source)
    compile(actual, name, 'exec')
    # Restore only authorized binding edits, then compare the entire AST.
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int:
                node.value = {338: 238, 331: 231, 162: 262}.get(node.value, node.value)
            elif isinstance(node.value, str):
                node.value = node.value.replace('338', '238').replace('331', '231').replace('162', '262').replace('epoch_337', 'epoch_237')
                for old, new in MODULES.items(): node.value = node.value.replace(new, old)
            return node
        def visit_ImportFrom(self, node):
            node.module = {v:k for k,v in MODULES.items()}.get(node.module, node.module)
            return node
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == '_cached_resource_api':
                return ast.ImportFrom(module='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                    names=[ast.alias(name='resource_api')], level=0)
            return self.generic_visit(node)
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


def test_dispatch_preserves_every_guard_and_requires_both_complete_proofs():
    source = Path('scripts/flow_matching/run_rcgan_full366_native_audit_dispatch_v1.py').read_text(encoding='utf-8')
    tree = dispatch_ast(source)
    text = ast.unparse(tree)
    assert 'saved_original_epochs=338' in text
    assert 'remaining_original_epochs=162' in text
    assert 'required_original_epochs=500' in text
    assert 'capped_wait' in text and 'if not passed:' in text
    assert 'resume_bind' in text
    with pytest.raises(ValueError): dispatch_ast('def main(): pass')


def test_incomplete_recovery_source_rejected():
    with pytest.raises(ValueError): recovery_ast('import sys')


def test_all_light_controllers_bind_without_scientific_imports_in_fresh_process():
    code = """import sys
from scripts.flow_matching.audit_rcgan_seed1105_chain_v1 import bind as chain
from scripts.flow_matching.audit_rcgan_seed1105_state_v1 import bind as state
from scripts.flow_matching.rcgan_seed1105_resume_runtime_v1 import bind as runtime
from scripts.flow_matching.run_rcgan_seed1105_recovery_v1 import bind as resume
from scripts.flow_matching.run_rcgan_seed1105_recovery_dispatch_v1 import bind as dispatch
for binding in (chain,state,runtime,resume,dispatch): binding()
assert 'numpy' not in sys.modules and 'tensorflow' not in sys.modules
"""
    subprocess.run([sys.executable, '-X', 'utf8', '-c', code], check=True)
