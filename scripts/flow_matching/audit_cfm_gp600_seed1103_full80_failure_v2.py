"""Single full80 unchanged-solver observation of saved GP600 seed1103 failure.

Reuse the full375000-update worker from seed1105. Change only the canonical
failed source identity and its actual14-of80 original scoring progress binding.
Never retrain, relax solver settings, repeat until pass or discard failed grids.
"""
import ast
import inspect
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as previous

ROOT,read,sha,write=previous.ROOT,previous.read,previous.sha,previous.write
MODULE='scripts.flow_matching.audit_cfm_gp600_seed1103_full80_failure_v2'
IDENTITY='cfm_ts_original__appendix_d__gp_paper_literal_e600__sine__seed1103'


def verify_ast():
    tree=ast.parse(inspect.getsource(previous.verify))
    class OriginalProgress(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if node.value==67:
                self.count+=1;node.value=14
            return node
    change=OriginalProgress();tree=change.visit(tree)
    if change.count!=1:raise ValueError('Only actual original scoring progress may change')
    return ast.fix_missing_locations(tree)


verify_values=dict(previous.__dict__,IDENTITY=IDENTITY)
exec(compile(verify_ast(),__file__,'exec'),verify_values)
verify=verify_values['verify']


def namespace():
    values=previous.namespace()
    values.update(verify=verify,__name__=MODULE)
    return values


def main():
    FunctionType(previous.previous.main.__code__,namespace())()


def post_audit(config):
    from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as arrays
    native=SimpleNamespace(ROOT=ROOT,read=read,sha=sha,verify=verify,validity_summary=previous.previous.validity_summary)
    return FunctionType(arrays.audit.__code__,dict(arrays.audit.__globals__,native=native))(config)


if __name__=='__main__':main()
