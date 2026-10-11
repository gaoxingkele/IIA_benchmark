"""Replay all original PhysioNet jobs with only a historical data-binding adapter."""
import ast
from pathlib import Path

from scripts.flow_matching import audit_saits_full_checkpoint_replay_v1 as original


def replay_ast():
    tree=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    class Bindings(ast.NodeTransformer):
        validator_count=0
        dispatch_count=0
        def visit_ImportFrom(self,n):
            if n.module=='scripts.flow_matching.validate_saits_full_result_v1':
                if [(a.name,a.asname) for a in n.names]!=[('validate',None)]:raise ValueError('Keep the complete validator interface')
                n.module='scripts.flow_matching.validate_saits_legacy_physio_full_result_v1';self.validator_count+=1
            return n
        def visit_Constant(self,n):
            if n.value=='scripts.flow_matching.audit_saits_full_checkpoint_replay_v1':
                n.value='scripts.flow_matching.audit_saits_physio_full_checkpoint_replay_v1';self.dispatch_count+=1
            return n
    adapter=Bindings();tree=adapter.visit(tree)
    if adapter.validator_count!=1 or adapter.dispatch_count!=1:
        raise ValueError('Only one validation import and the isolated worker dispatch may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(__name__='_isolated_saits_physio_native_full_replay',__file__=original.__file__)
    exec(compile(replay_ast(),original.__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
