"""Same cohort/seed audit with the reviewed worker identity and recovery queue."""
import ast
import argparse
import json
from pathlib import Path
from scripts.flow_matching import summarize_grasp_protocol as original


def summarize(queue,root=original.ROOT):
    source = Path(original.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
    count=0
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,node):
            nonlocal count
            if node.value=='scripts.flow_matching.grasp_protocol_runner':
                count+=1
                return ast.copy_location(ast.Constant('scripts.flow_matching.grasp_protocol_cuda_runtime'),node)
            return node
    function=Binding().visit(function)
    if count!=1:
        raise ValueError('Original live-process audit changed')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['summarize'](queue,root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',default='configs/experiments/fm_grasp_protocol_queue.v3.json')
    parser.add_argument('--output',default='docs/reports/fm_grasp_protocol_progress_2026-10-09_v3.json')
    args=parser.parse_args()
    queue=json.loads((original.ROOT/args.queue).read_text(encoding='utf-8'))
    report=summarize(queue)
    report.update(queue=args.queue,queue_sha256=original.sha(original.ROOT/args.queue))
    original.write_json(original.ROOT/args.output,report)
    print(json.dumps({k:report[k] for k in ['entity_job_counts','registered_entity_jobs','registered_cohorts','completed_cohorts','active_process_verified_live']}))


if __name__=='__main__':
    main()
