"""One full20 original BB600/9600-update saved-model failure observation.

Keep the established original native20 solver and independent-grid auditor.
Only actual failed-source identity, BB method, optimizer budget and original
11/20 failure progress change. CFM-TS: OpenReview Hqn4Aj7xrQ.
"""
import ast
import inspect
from types import FunctionType
from scripts.flow_matching import audit_cfm_main_GP6001104_full20_failure_v1 as previous

ROOT,read,sha,write=previous.ROOT,previous.read,previous.sha,previous.write
MODULE='scripts.flow_matching.audit_cfm_main_BB6001106_full20_failure_v1'
IDENTITY='cfm_ts_original__main_text__bb_paper_literal_e600__sine__seed1106'


def verify_ast():
    tree=ast.parse(inspect.getsource(previous.verify));replacements={93600:9600,'gp':'bb',12:11};counts={k:0 for k in replacements}
    class ActualFailure(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value in replacements:counts[node.value]+=1;node.value=replacements[node.value]
            return node
    tree=ActualFailure().visit(tree)
    if counts!={k:1 for k in replacements}:raise ValueError('Only actual original method/budget/progress may change')
    return ast.fix_missing_locations(tree)


values=dict(previous.__dict__,IDENTITY=IDENTITY)
exec(compile(verify_ast(),__file__,'exec'),values)
verify=values['verify']


def namespace():
    values=previous.namespace();values.update(verify=verify,__name__=MODULE,__doc__=__doc__)
    return values


def post_audit(config):
    return FunctionType(previous.post_audit.__code__,dict(previous.post_audit.__globals__,verify=verify))(config)


def main():namespace()['main']()


if __name__=='__main__':main()
