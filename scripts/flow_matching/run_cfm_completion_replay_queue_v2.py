"""Retain all285 full CFM replays and their85 actual certificates after routing.

The original complete model/grid worker and independent review are unchanged.
Only a verified idle controller is handed off, and telemetry is cached.
"""
import ast
import inspect
from types import FunctionType

from scripts.flow_matching import run_cfm_completion_replay_queue_v1 as original


def cached_resource_api():
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api=resource_api();api['resource_snapshot']=resource_snapshot
    return api


def allowed_idle(state,descendants):
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child
    return (state.get('active_pid') is None and state.get('active_job') is None
        and state.get('required_original_models')==285 and state.get('pending_original_models')==200
        and len(state.get('verified_original_ids',[]))==85 and not state.get('failed_preserved')
        and state.get('state')=='waiting_actual_original_training_completion'
        and all(allowed_console_child(p['name'],p['exe']) for p in descendants))


def verified_handoff(runtime,base,update):
    from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as common
    def phase(role,state,output_exists,descendants):
        return allowed_idle(state,descendants)
    namespace=dict(common.__dict__,handoff_phase_allowed=phase)
    handoff=FunctionType(common.safe_handoff.__code__,namespace)
    route=dict(previous_owner=runtime['previous_owner'],previous_state_path=runtime['previous_state_path'],
        runtime_configuration=runtime['runtime_configuration'])
    handoff('original',route,base,update,runtime['poll_seconds'])


def candidate_ready(route):
    if not (original.ROOT/route['result_path']).is_file():
        return False
    if 'checkpointed_training_terminal' not in route:
        return True
    path=original.ROOT/route['checkpointed_training_terminal']
    if not path.is_file():
        return False
    terminal=original.read(path)
    return (terminal['exit_code']==0 and terminal['complete']
        and terminal['full_original400_checkpointed_result_verified']
        and not terminal['resource_observations']['resource_guard_aborted'])


def controller_ast():
    tree=ast.parse(inspect.getsource(original.controller));function=tree.body[0]
    assignments=[n for n in ast.walk(function) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='resource_api']
    if len(assignments)!=1:raise ValueError('Only original host telemetry binding may change')
    assignments[0].func.id='_cached_resource_api'
    class CandidateGate(ast.NodeTransformer):
        matches=0
        def visit_UnaryOp(self,node):
            if ast.dump(node)==ast.dump(ast.parse("not (ROOT/route['result_path']).is_file()",mode='eval').body):
                self.matches+=1
                return ast.parse('not _candidate_ready(route)',mode='eval').body
            return self.generic_visit(node)
    gate=CandidateGate();tree=gate.visit(tree)
    if gate.matches!=1:raise ValueError('Only the source candidate file check may require the new actual producer terminal')
    tries=[n for n in function.body if isinstance(n,ast.Try) and n.body and isinstance(n.body[0],ast.While)]
    if len(tries)!=1:raise ValueError('Exactly the full original285 controller loop is required')
    tries[0].body.insert(0,ast.parse('_verified_handoff(runtime,base,update)').body[0])
    # Native workers remain the same auditor/reviewer modules. No math route,
    #result tolerance, grid, checkpoint count or existing85 proof is altered.
    return ast.fix_missing_locations(tree)


def namespace():
    values=dict(original.__dict__,_cached_resource_api=cached_resource_api,_verified_handoff=verified_handoff,
        _candidate_ready=candidate_ready)
    for name,value in original.__dict__.items():
        if isinstance(value,FunctionType) and value.__module__==original.__name__:
            rebound=FunctionType(value.__code__,values,value.__name__,value.__defaults__,value.__closure__)
            rebound.__kwdefaults__=value.__kwdefaults__;values[name]=rebound
    exec(compile(controller_ast(),__file__,'exec'),values)
    return values


if __name__=='__main__':
    namespace()['main']()
