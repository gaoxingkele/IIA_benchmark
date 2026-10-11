"""Keep the full native replay followup lightweight while its producer works.

Bind only scheduler resource functions using the existing AST loader. The frozen
v1 verification, full-coverage gates and actual child command remain unchanged.
"""
import ast
from pathlib import Path

from scripts.flow_matching import followup_giflow_seed_replay_verification_v1 as original

def bound_main_tree(source):
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    replaced=0
    body=[]
    modules={'scripts.flow_matching.heavy_resources','scripts.flow_matching.heavy_scheduler_v2'}
    for node in main.body:
        if isinstance(node,ast.ImportFrom) and node.module in modules:
            replaced+=1
            continue
        body.append(node)
        if isinstance(node,ast.ImportFrom) and node.module=='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
            body.extend(ast.parse("api=resource_api()\nexclusive_lock=api['exclusive_lock']\nresource_slot=api['resource_slot']").body)
    if replaced!=2:
        raise ValueError('Exactly the two heavyweight resource imports must be replaced')
    main.body=body
    return ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))

def main():
    source=Path(original.__file__).read_text(encoding='utf-8')
    namespace=dict(vars(original))
    exec(compile(bound_main_tree(source),original.__file__,'exec'),namespace)
    namespace['main']()

if __name__=='__main__':
    main()
