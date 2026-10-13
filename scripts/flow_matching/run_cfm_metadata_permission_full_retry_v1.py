"""Original full285 controller, selecting two unchanged full400 metadata retries.

Keep the original family lock and all resource limits. Only the isolated worker
entry selects the metadata-safe wrapper; frozen native numerical code is reused.
"""
import ast
import inspect
import argparse
import sys
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as original
from scripts.flow_matching.run_cfm_streaming_idle_handoff_v3 import streaming_sha
from scripts.flow_matching.run_cfm_metadata_safe_original_job_v1 import safe_write

ROOT=original.ROOT


def verify_runtime(runtime):
    canonical=original.read(ROOT/runtime['controllers']['original']['queue'])
    selected,queue=original.binding(runtime,'recovery')
    old={j['id']:j for j in canonical['jobs']}
    chosen=[j for j in queue['jobs'] if j['id'] in selected['selected_job_ids']]
    required=['cfm_ts_original__main_text__bb_paper_literal_e400__sine__seed'+str(s) for s in (1104,1106)]
    if len(old)!=285 or len(queue['jobs'])!=285 or queue['resource_policy']!=canonical['resource_policy']:
        raise ValueError('Every285 original slot and all original resource safeguards required')
    if [j.get('recovery_original_job_id') for j in chosen]!=required:
        raise ValueError('Exactly both unsaved metadata failures must be selected')
    from scripts.flow_matching.run_cfm_completion_replay_queue_v1 import check_binding
    identities=[]
    for job in queue['jobs']:
        identity=job.get('recovery_original_job_id',job['id']);check_binding(old[identity],job);identities.append(identity)
    if len(set(identities))!=285:raise ValueError('Canonical slot duplicated or missing')
    for job in chosen:
        identity=job['recovery_original_job_id'];source=old[identity]
        receipt=original.read(ROOT/canonical['state_root']/'receipts'/(identity+'.json'))
        progress=original.read(ROOT/source['output_directory']/'progress.json')
        stderr=(ROOT/canonical['state_root']/'logs'/(identity+'.stderr.log')).read_text(encoding='utf-8')
        if (receipt['exit_code']!=1 or receipt['complete'] or receipt['resource_observations']['resource_guard_aborted']
                or 'PermissionError: [WinError 5]' not in stderr or progress['phase']!='training'
                or progress['epoch'] not in (61,313) or (Path(source['artifact_directory'])/'model.pt').exists()
                or (ROOT/source['output_directory']/'result.json').exists()
                or job['epochs']!=400 or job['expected_optimizer_updates']!=6400
                or job['output_directory']==source['output_directory'] or job['artifact_directory']==source['artifact_directory']):
            raise ValueError('Only original unsaved metadata failures may receive fresh complete training')
    return chosen,queue


def main_ast():
    tree=ast.parse(inspect.getsource(original.main))
    class Dispatch(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if node.value=='scripts.flow_matching.run_cfm_ts_original_job':
                self.count+=1;node.value='scripts.flow_matching.run_cfm_metadata_safe_original_job_v1'
            return node
    change=Dispatch();tree=change.visit(tree)
    if change.count!=1:raise ValueError('Only one isolated numerical entry point may change')
    return ast.fix_missing_locations(tree)


def cached_api():
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api=original.resource_api();api['resource_snapshot']=resource_snapshot
    return api


def namespace():
    values=dict(original.__dict__,sha=streaming_sha,write=safe_write,resource_api=cached_api)
    for name,fn in original.__dict__.items():
        if isinstance(fn,FunctionType) and fn.__module__==original.__name__:
            values[name]=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
    values.update(sha=streaming_sha,write=safe_write,resource_api=cached_api)
    exec(compile(main_ast(),__file__,'exec'),values)
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--runtime-config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();runtime=original.read(ROOT/args.runtime_config);chosen,queue=verify_runtime(runtime)
    if args.probe_only:
        print(dict(full_original_slots=285,selected_complete_retries=len(chosen),epochs=[j['epochs'] for j in chosen],
            original_optimizer_updates=[j['expected_optimizer_updates'] for j in chosen],original_guards_unchanged=True,
            partial_epoch_credit=0,training_executed=False,torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules));return
    saved=sys.argv
    try:
        sys.argv=[saved[0],'--runtime-config',args.runtime_config,'--controller','recovery']
        namespace()['main']()
    finally:sys.argv=saved


if __name__=='__main__':main()
