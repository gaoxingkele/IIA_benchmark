"""Yield long enough for native waiters after each unchanged full GRASP job."""
import argparse
import ast
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_grasp_protocol_queue as original


def bound_main(yield_seconds):
    if not isinstance(yield_seconds,int) or yield_seconds<30:
        raise ValueError('Yield must cover the registered native polling interval')
    source=Path(original.__file__);tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    changes={'configs/experiments/fm_grasp_protocol_queue.v2.json':'configs/experiments/fm_grasp_protocol_queue.v4.json',
        'scripts.flow_matching.grasp_protocol_runner':'scripts.flow_matching.grasp_protocol_cuda_runtime'}
    counts=dict.fromkeys(changes,0);yields=0
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in changes:
                counts[node.value]+=1;return ast.copy_location(ast.Constant(changes[node.value]),node)
            return node
        def visit_With(self,node):
            nonlocal yields
            node=self.generic_visit(node)
            if (len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call)
                    and isinstance(node.items[0].context_expr.func,ast.Subscript)
                    and isinstance(node.items[0].context_expr.func.slice,ast.Constant)
                    and node.items[0].context_expr.func.slice.value=='resource_slot'):
                yields+=1
                return [node,ast.parse('time.sleep('+str(yield_seconds)+')').body[0]]
            return node
    function=Binding().visit(function)
    if any(count!=1 for count in counts.values()) or yields!=1:
        raise ValueError('Original full guarded orchestration changed')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['main']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    runtime=json.loads((original.ROOT/args.runtime_config).read_text(encoding='utf-8'))
    for entry in runtime['source_receipts']:
        if original.sha(original.ROOT/entry['path'])!=entry['sha256']:raise ValueError('Frozen native fairness source differs')
    path=original.ROOT/runtime['queue'];queue=json.loads(path.read_text(encoding='utf-8'))
    if original.sha(path)!=runtime['queue_sha256']:raise ValueError('All original full native jobs changed')
    original.verify(queue);function=bound_main(runtime['yield_seconds'])
    if args.probe_only:
        print(json.dumps(dict(full_jobs=len(queue['jobs']),cohorts=queue['cohort_count'],
            post_job_yield_seconds=runtime['yield_seconds'],full_scientific_queue_unchanged=True)));return
    sys.argv=['original-full-grasp','--queue',str(path)];function()


if __name__=='__main__':main()
