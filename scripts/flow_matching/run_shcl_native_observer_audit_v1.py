"""Apply the unchanged native operator audit caps to observer fidelity plumbing."""
import argparse
import ast
import inspect
from types import FunctionType

from scripts.flow_matching import run_shcl_author_audits_v1 as original


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--config',required=True)
    args=parser.parse_args(); config=original.native.read(original.native.ROOT/args.config)
    original.verify(config)
    tree=ast.parse(inspect.getsource(original.main)); changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.shcl_author_audits_v1':
            node.value='scripts.flow_matching.audit_shcl_native_observer_v1'; changes+=1
        elif isinstance(node,ast.Constant) and node.value=='full_released_operator_audit.json':
            node.value='full_observer_fidelity_fixture.json'; changes+=1
    # Adapt only the expected diagnostic report key; all dispatch guards remain.
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and ast.dump(node)==ast.dump(ast.parse(
                "values.get('full_operator_profiles')!=13",mode='eval').body)):
            node.left=ast.parse("values['all_native_parameters_and_Adam_state_bitwise_unchanged_by_observation']",mode='eval').body
            node.ops=[ast.IsNot()]; node.comparators=[ast.Constant(value=True)]; changes+=1
    if changes!=3: raise ValueError('Only the fresh fixture entry/report contract may change')
    namespace=dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace); namespace['main']()


if __name__=='__main__':main()
