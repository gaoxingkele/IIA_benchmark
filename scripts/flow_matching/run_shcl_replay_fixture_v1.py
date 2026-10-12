"""Keep all original observer replay caps and change only the fresh fixture entry."""
import ast
import inspect

from scripts.flow_matching import run_shcl_native_observer_audit_v1 as observer_guard
from scripts.flow_matching import run_shcl_author_audits_v1 as original


def main():
    tree=ast.parse(inspect.getsource(observer_guard.main)); changes=0
    # Preserve the existing audited compiler and all its cap/slot operations.
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.audit_shcl_native_observer_v1':
            node.value='scripts.flow_matching.audit_shcl_replay_fixture_v1'; changes+=1
    if changes!=1: raise ValueError('Exactly one isolated fixture entrypoint changes')
    namespace=dict(observer_guard.__dict__); exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
