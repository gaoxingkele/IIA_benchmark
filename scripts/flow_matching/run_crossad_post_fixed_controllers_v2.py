"""Keep full139/four recovery obligations and route post-review to its API fix.

Only idle controllers can be handed off. Native training/model replay, raw
point review, source code, batches and numerical/resource limits are retained.
"""
import argparse
import ast
import inspect
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as original
from scripts.flow_matching import run_crossad_recovery_completion_handoff_v1 as recovery

POST_MODULE='scripts.flow_matching.replay_crossad_post_metric_keywords_v2'


def controller_ast(role):
    module=original if role=='original' else recovery
    tree=ast.parse(inspect.getsource(module.controller));changed=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or getattr(node.func,'id',None)!='run_stage':continue
        position=1 if role=='original' else 2
        if (len(node.args)>position+1 and isinstance(node.args[position+1],ast.Constant)
                and node.args[position+1].value=='post'):
            if getattr(node.args[position],'value',None)!='scripts.flow_matching.run_crossad_completion_replay_queue_v1':
                raise ValueError('Keep the original full independent post route as the only replacement')
            node.args[position].value=POST_MODULE;changed+=1
    if changed!=1:raise ValueError('Require exactly one complete original post-review dispatch site')
    if role=='recovery':
        loops=[n for n in ast.walk(tree) if isinstance(n,ast.While) and ast.unparse(n.test)=='len(covered) < 139']
        if len(loops)!=1:raise ValueError('Keep the entire original139 canonical union loop')
        loops[0].body.insert(0,ast.parse('_adopt_external_recovery(config,jobs,routes,recovered,covered)').body[0])
    return ast.fix_missing_locations(tree)


def adopt_external_recovery(config,jobs,routes,recovered,covered):
    for candidate in config.get('external_completed_recoveries',[]):
        identity=candidate['canonical_id']
        if identity in covered:continue
        if identity not in routes:raise ValueError('External evidence must bind an existing exact recovery obligation')
        post_path=candidate['post_review_configuration'];post=original.read(post_path)
        receipt_path=original.ROOT/post['state_root']/'receipt.json'
        if not receipt_path.exists():continue
        receipt=original.read(receipt_path)
        if receipt['exit_code']!=0 or receipt['resource_guard_aborted'] or not receipt['independent_stage_passed']:
            continue  # A failed external attempt remains failed; other routes continue.
        route,job=routes[identity]
        base=original.read(config['original_completion_configuration'])
        recovery.exact_route(route,jobs[identity],original.read(route['native_queue']),original.read(base['original_queue']))
        certificate=recovery.completed_review(post_path,job,config)
        evidence=dict(certificate,canonical_id=identity,recovery_job_id=job['id'],native_queue=route['native_queue'],
            completion_origin='exact_external_native_and_keyword_corrected_independent_recovery',additional_training_seed_slots=0)
        original.exclusive_json(original.ROOT/config['state_root']/'completed'/(identity+'.json'),evidence)
        recovered[identity]=evidence;covered[identity]=evidence


def allowed_idle(role,state,output_exists,descendants):
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child
    phases={'original':'waiting_actual_original_full_execution_completion',
        'recovery':'waiting_actual_exact_recovery_and_original_full139_completion'}
    return (role in phases and state.get('state')==phases[role] and state.get('active_pid') is None
        and len(state.get('verified_original_ids',[]))<139 and not output_exists
        and all(allowed_console_child(p['name'],p['exe']) for p in descendants))


def cached_resource_api():
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api=original.resource_api();api['resource_snapshot']=resource_snapshot
    return api


def live_stage_jobs():
    ids=original.live_stage_jobs()
    prefixes=('replay_crossad_post_metric_keywords','replay_crossad_attention_lifetime','replay_crossad_residual_lifetime')
    for process in original.psutil.process_iter(['name']):
        if 'python' not in (process.info['name'] or '').lower():continue
        try:
            command=process.cmdline();created=process.create_time()
            if '--config' not in command or not any(any(tag in part for tag in prefixes) for part in command):continue
            config=original.read(command[command.index('--config')+1])
            identity=config.get('job_id')
            if identity is None and config.get('phase')=='post':identity=original.read(config['native_configuration'])['job_id']
            if identity and process.create_time()==created and process.cmdline()==command:ids.add(identity)
        except (original.psutil.NoSuchProcess,original.psutil.AccessDenied,OSError,ValueError,KeyError):continue
    return ids


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--handoff-config',required=True);parser.add_argument('--role',choices=['original','recovery'],required=True)
    args=parser.parse_args();handoff=original.read(args.handoff_config)
    for pin in handoff['source_receipts']:
        if original.digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen full controller post-route binding changed')
    selected=handoff['roles'][args.role];path=selected['controller_configuration'];config=original.read(path)
    module=original if args.role=='original' else recovery
    if args.role=='original':original.verify(config)
    else:recovery.verify(config)
    from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as common
    safe_handoff=FunctionType(common.safe_handoff.__code__,dict(common.__dict__,handoff_phase_allowed=allowed_idle))
    base=original.ROOT/selected['handoff_state_root'];base.mkdir(parents=True,exist_ok=True)
    def update(values):original.write(base/'status.json',values)
    safe_handoff(args.role,dict(previous_owner=selected['previous_owner'],previous_state_path=selected['previous_state_path'],
        runtime_configuration=args.handoff_config,output_directory=selected['completion_output']),base,update,10)
    namespace=dict(module.__dict__,_adopt_external_recovery=adopt_external_recovery)
    if args.role=='original':namespace.update(resource_api=cached_resource_api,live_stage_jobs=live_stage_jobs)
    else:namespace['original']=SimpleNamespace(**dict(original.__dict__,live_stage_jobs=live_stage_jobs))
    exec(compile(controller_ast(args.role),__file__,'exec'),namespace)
    namespace['controller'](config,path)


if __name__=='__main__':main()
