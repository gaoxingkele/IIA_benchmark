"""Full point/native/independent replay of completed causal-axis PSM seed42.

Keep all1140 original jobs/six seeds and the original full score streams,
guards and tolerance. No retraining or TAB/journal equivalence certification.
"""
import ast
import inspect
from types import FunctionType
from scripts.flow_matching import run_pi_source1105_completed_full_replay_v1 as previous

ROOT,read,digest,SCOPE=previous.ROOT,previous.read,previous.digest,previous.SCOPE
CANONICAL='pi_author__causal_axis_repaired__psm__full__seed42'
RECOVERY=CANONICAL+'__full_incident_recovery_v5'


def verify_ast():
    tree=ast.parse(inspect.getsource(previous.verify))
    required={"job['seed'] != 1105":42,"job['branch'] != 'source_runtime'":'causal_axis_repaired'}
    seen={key:0 for key in required}
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare):
            key=ast.unparse(node)
            if key in required:
                node.comparators=[ast.Constant(required[key])];seen[key]+=1
    if any(n!=1 for n in seen.values()):
        raise ValueError('Change only the selected full original seed and branch bindings')
    return ast.fix_missing_locations(tree)


def namespace():
    values=dict(previous.__dict__,CANONICAL=CANONICAL,RECOVERY=RECOVERY,__doc__=__doc__)
    exec(compile(verify_ast(),__file__,'exec'),values)
    return values


def verify(config):return namespace()['verify'](config)


def main():FunctionType(previous.main.__code__,namespace())()


if __name__=='__main__':main()
