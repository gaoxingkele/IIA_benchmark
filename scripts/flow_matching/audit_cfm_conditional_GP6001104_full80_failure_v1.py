"""Observe all80 original conditional GP600 seed1104 grids once after underflow.

Restore the full375000-update checkpoint without fitting; preserve solver,
dtype, every grid, guards and explicit failures. No successful-subset benchmark.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import ast
import inspect
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as previous

ROOT,read,sha,write=previous.ROOT,previous.read,previous.sha,previous.write
MODULE='scripts.flow_matching.audit_cfm_conditional_GP6001104_full80_failure_v1'
IDENTITY='cfm_ts_original__appendix_d__gp_gaussian_conditional_e600__sine__seed1104'


def verify_ast():
    tree=ast.parse(inspect.getsource(previous.verify));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare) and ast.unparse(node)=="progress['test_trajectories_completed'] != 67":
            node.comparators=[ast.Constant(58)];changed+=1
    if changed!=1:raise ValueError('Only the observed original scoring progress binding may change')
    return ast.fix_missing_locations(tree)


def verify(config):
    values=dict(previous.__dict__,IDENTITY=IDENTITY)
    exec(compile(verify_ast(),__file__,'exec'),values)
    job,model=values['verify'](config)
    if job['variant']!='gp_gaussian_conditional' or model['parameters']['target_mode']!='gaussian_conditional':
        raise ValueError('The actual failed conditional GP target must remain unchanged')
    return job,model


def namespace():
    values=previous.namespace()
    values.update(verify=verify,__name__=MODULE)
    return values


def main():FunctionType(previous.previous.main.__code__,namespace())()


def post_audit(config):
    from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as arrays
    native=SimpleNamespace(ROOT=ROOT,read=read,sha=sha,verify=verify,validity_summary=previous.previous.validity_summary)
    return FunctionType(arrays.audit.__code__,dict(arrays.audit.__globals__,native=native))(config)


if __name__=='__main__':main()
