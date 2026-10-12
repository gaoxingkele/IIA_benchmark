"""Correct only math.isclose keyword names in the full independent review."""
import argparse
import ast
import inspect
from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as original


def math_tolerance():
    return dict(rel_tol=original.METRIC_TOLERANCE['rtol'],abs_tol=original.METRIC_TOLERANCE['atol'])


def post_review_ast():
    tree=ast.parse(inspect.getsource(original.post_review));changes=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or ast.unparse(node.func)!='math.isclose':continue
        for keyword in node.keywords:
            if keyword.arg is None and ast.dump(keyword.value)==ast.dump(ast.Name(id='METRIC_TOLERANCE',ctx=ast.Load())):
                keyword.value=ast.parse('_math_tolerance()',mode='eval').body;changes+=1
    if changes!=2:raise ValueError('Exactly the original two scalar comparison keyword bindings must change')
    return ast.fix_missing_locations(tree)


def stage_controller_ast():
    tree=ast.parse(inspect.getsource(original.stage_controller));changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_crossad_completion_replay_queue_v1':
            node.value='scripts.flow_matching.replay_crossad_post_metric_keywords_v2';changes+=1
    if changes!=1:raise ValueError('Only the fresh independent worker route may change')
    return tree


def cached_resource_api():
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api=original.resource_api();api['resource_snapshot']=resource_snapshot
    return api


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    parser.add_argument('--stage-worker',action='store_true');parser.add_argument('--stage-controller',action='store_true')
    args=parser.parse_args();config=original.read(args.config)
    if config['phase']!='post':raise ValueError('Only full independent post-review is admitted; never replay the native model')
    for pin in config['source_receipts']:
        if original.digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen exact independent review source changed')
    namespace=dict(original.__dict__,_math_tolerance=math_tolerance,resource_api=cached_resource_api)
    exec(compile(post_review_ast(),__file__,'exec'),namespace)
    exec(compile(ast.parse(inspect.getsource(original.stage_worker)),__file__,'exec'),namespace)
    exec(compile(stage_controller_ast(),__file__,'exec'),namespace)
    namespace['stage_worker' if args.stage_worker else 'stage_controller'](config,args.config)


if __name__=='__main__':main()
