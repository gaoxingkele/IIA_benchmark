"""Run unchanged registered CPU baselines in an independently bounded serial lane."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
from types import FunctionType

from scripts.flow_matching import run_resource_gated_light_controller_v1 as original
from scripts.flow_matching.run_light_controller import ROOT, digest


def validate_limits(before, after):
    for key in ('minimum_available_memory_bytes','minimum_commit_headroom_bytes','emergency_commit_headroom_bytes'):
        if after[key] < before[key]:
            raise ValueError('Every original physical and commit safeguard must remain or strengthen')
    if (after['maximum_worker_rss_bytes'] <= 0 or after['maximum_worker_private_bytes'] < after['maximum_worker_rss_bytes']
            or after['minimum_commit_headroom_bytes'] < after['emergency_commit_headroom_bytes'] + after['maximum_worker_private_bytes']):
        raise ValueError('Full private cap must fit above the unchanged emergency reserve')
    for key in ('guard_poll_seconds','controller_yield_seconds'):
        if after[key] != before[key]:
            raise ValueError('Original poll and yield cadence required')


def capped_class_ast(source):
    tree=ast.parse(source)
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GuardedChild'))
    method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='poll')
    conditions=[n for n in ast.walk(method) if isinstance(n,ast.If) and ast.unparse(n.test)=="available < self.settings['emergency_commit_headroom_bytes']"]
    if len(conditions)!=1:raise ValueError('Exactly the original emergency guard must remain identifiable')
    conditions[0].test=ast.BoolOp(op=ast.Or(),values=[conditions[0].test,
        ast.parse("self.peaks['peak_tree_rss_bytes'] > self.settings['maximum_worker_rss_bytes']",mode='eval').body,
        ast.parse("self.peaks['peak_tree_private_bytes'] > self.settings['maximum_worker_private_bytes']",mode='eval').body])
    return ast.fix_missing_locations(node)


def bind(registration):
    parent=ROOT/registration['parent_resource_registration']
    if digest(parent)!=registration['parent_resource_registration_sha256']:
        raise ValueError('Original complete resource adapter registration changed')
    before=json.loads(parent.read_text(encoding='utf-8'))
    validate_limits(before['settings'],registration['settings'])
    if (registration['original_runtime']!=before['original_runtime'] or set(registration['controllers'])!={'strict_cpu'}
            or registration['controllers']['strict_cpu']!=before['controllers']['strict_cpu']
            or registration['resource_lock']==before['resource_lock']):
        raise ValueError('Only the entire original strict CPU queue may use this distinct serial lane')
    for source in registration['source_receipts']:
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Frozen bounded CPU orchestration source changed')
    selected=registration['controllers']['strict_cpu']
    queue=json.loads((ROOT/selected['queue']).read_text(encoding='utf-8'))
    if (digest(ROOT/selected['queue'])!=selected['queue_sha256'] or len(queue['jobs'])!=175
            or queue['evaluation']['cpu_threads']!=2 or queue['lane']!='cpu_baselines'
            or any(j['parameter_overrides']['device']!='cpu' for j in queue['jobs'])):
        raise ValueError('All175 original CPU baseline jobs and their2threads required')
    namespace=dict(vars(original))
    source=(ROOT/'scripts/flow_matching/run_resource_gated_light_controller_v1.py').read_text(encoding='utf-8')
    node=capped_class_ast(source)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'scripts/flow_matching/run_resource_gated_light_controller_v1.py'),'exec'),namespace)
    namespace['guarded_factory']=FunctionType(original.guarded_factory.__code__,namespace)
    return FunctionType(original.main.__code__,namespace)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    registration=json.loads((ROOT/args.runtime_config).read_text(encoding='utf-8'))
    function=bind(registration)
    sys.argv=['bounded-original-strict-cpu','--runtime-config',args.runtime_config,'--controller','strict_cpu']+(['--probe-only'] if args.probe_only else [])
    function()


if __name__=='__main__':main()
