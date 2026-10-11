"""Run one stopped canonical MaelNet attempt through its unchanged full pipeline.

Only the author's existing --job-id selector is passed to the existing guarded
stage controller. The complete150-job registry and all native stages remain.
"""
import ast
import copy
from pathlib import Path

from scripts.flow_matching import run_maelnet_stage_resource_queue_v1 as original


def compare_recovery(original_queue, recovery_queue, original_id, attempt_id):
    if len(original_queue['jobs'])!=150 or len(recovery_queue['jobs'])!=150:
        raise ValueError('Keep every150 original MaelNet canonical slots')
    old_settings={k:v for k,v in original_queue['settings'].items() if k!='output_root'}
    new_settings={k:v for k,v in recovery_queue['settings'].items() if k!='output_root'}
    if old_settings!=new_settings:
        raise ValueError('Original author environment and scientific settings must stay exact')
    changed=0
    for old,new in zip(original_queue['jobs'],recovery_queue['jobs']):
        if old['id']!=original_id:
            if old!=new:
                raise ValueError('All other149 original jobs must stay exact')
            continue
        changed+=1
        if (new['id']!=attempt_id or new.get('recovery_original_job_id')!=original_id
                or new['output_directory']==old['output_directory']
                or {k:v for k,v in new.items() if k not in ('id','output_directory','recovery_original_job_id')}
                != {k:v for k,v in old.items() if k not in ('id','output_directory')}):
            raise ValueError('Only isolated attempt identity/output may change; full stages/data/model/seed retained')
    if changed!=1 or len({j['id'] for j in recovery_queue['jobs']})!=150:
        raise ValueError('Exactly one unique original canonical attempt must receive recovery')


def selected_main_ast():
    tree=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    changes=0
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                and t.value.id=='sys' and t.attr=='argv' for t in node.targets):
            node.value=ast.BinOp(left=node.value,op=ast.Add(),
                right=ast.parse("['--job-id', registration['selected_job_id']]",mode='eval').body)
            changes+=1
    if changes!=1:
        raise ValueError('Exactly the original existing author job selector must be bound')
    return ast.fix_missing_locations(function)


def main():
    # Existing runtime/queue/source verification, original stage workers, full
    # author functions and GuardedChild are inherited without modification.
    namespace=dict(original.__dict__)
    function=selected_main_ast()
    exec(compile(ast.Module(body=[function],type_ignores=[]),original.__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':
    main()
