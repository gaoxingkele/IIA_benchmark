"""Link all eight new full-budget producers to the original complete285 replay.

The existing live full285 dispatcher remains. This supplement dispatches only
the eight new execution routes, retains every other canonical slot, and adopts
actual successful native-plus-independent certificates from the live primary.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import ast
import inspect
from types import FunctionType

from scripts.flow_matching import run_cfm_completion_replay_queue_v1 as original
from scripts.flow_matching import run_cfm_uncovered_resource_aborts_v1 as producer

ROOT=original.ROOT
PREVIOUS='configs/runtime/fm_cfm_completion_replay_queue.v4.json'
PRODUCER='configs/runtime/fm_cfm_all8_uncovered_resource_abort_recovery.v1.json'


def expected_routes(previous,recovery):
    jobs={j.get('recovery_original_job_id',j['id']):j for j in recovery['jobs']}
    chosen=set(recovery['recovery_job_ids'])
    return [dict(route,job=jobs[route['original_canonical_id']],
        result_path=jobs[route['original_canonical_id']]['output_directory']+'/result.json')
        if jobs[route['original_canonical_id']]['id'] in chosen else route for route in previous['routes']]


def verify(runtime):
    previous=original.read(ROOT/PREVIOUS);production=original.read(ROOT/PRODUCER)
    selected,recovery=producer.verify(production)
    if (runtime['previous_full285_configuration']!=PREVIOUS or runtime['producer_configuration']!=PRODUCER
        or runtime['routes']!=expected_routes(previous,recovery)
        or runtime['selected_canonical_ids']!=[j['recovery_original_job_id'] for j in selected]):
        raise ValueError('Every285 exact route retained; only all8 new source executions may change')
    for key in ['python','cpu_threads','settings','resource_lock','environment_lock','prediction_tolerance','original_scope']:
        if runtime[key]!=previous[key]:raise ValueError('Original native full solver/review guard changed: '+key)
    routes,template,verified=original.verify(runtime)
    adopt_primary(runtime,verified)
    return routes,template,verified


def adopt_primary(runtime,verified):
    primary=original.read(ROOT/PREVIOUS)
    path=ROOT/primary['state_root']/'verified_full_model_records.json'
    if not path.exists():return
    records=original.read(path)['records']
    routes={r['original_canonical_id']:r for r in primary['routes']}
    for record in records:
        identity=record['original_canonical_id']
        if identity in verified:
            if verified[identity]!=record:raise ValueError('Conflicting immutable primary certificate')
            continue
        if identity in runtime['selected_canonical_ids']:
            raise ValueError('New retry slot unexpectedly already completed by another execution')
        nativepath=ROOT/primary['child_config_root']/(identity+'.native.json')
        native=original.read(nativepath)
        report=original.read(ROOT/native['output_directory']/'checkpoint_replay_audit.json')
        receipt=original.read(ROOT/native['state_root']/'receipt.json')
        original.check_native_terminal(native,report,receipt)
        reviewpath=ROOT/primary['child_config_root']/(identity+'.review.json')
        review=original.read(reviewpath)
        proof=original.read(ROOT/review['output_proof'])
        guard=original.read(ROOT/review['state_root']/'receipt.json')
        if (not proof['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or not guard['independent_full_array_review_passed']
            or proof['configuration_sha256']!=original.sha(reviewpath)
            or proof['records']!=[record]):
            raise ValueError('Real guarded full primary certificate required')
        route=routes[identity];canonical=dict(route,result_sha256=original.sha(ROOT/route['result_path']))
        checked=original.verify_record(native,canonical,record)
        if checked!=record:raise ValueError('Every primary full-coordinate proof must remain exact')
        verified[identity]=record


def selected_ready(route,runtime):
    return route['original_canonical_id'] in runtime['selected_canonical_ids'] and (ROOT/route['result_path']).is_file()


def controller_ast():
    tree=ast.parse(inspect.getsource(original.controller));reference=ast.dump(ast.parse("not (ROOT/route['result_path']).is_file()",mode='eval').body)
    class Gate(ast.NodeTransformer):
        count=0
        def visit_UnaryOp(self,node):
            if ast.dump(node)==reference:
                self.count+=1;return ast.parse('not _selected_ready(route,runtime)',mode='eval').body
            return self.generic_visit(node)
    gate=Gate();tree=gate.visit(tree)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.While) and isinstance(n.test,ast.Constant) and n.test.value is True]
    if gate.count!=1 or len(loops)!=1:raise ValueError('Only selected candidate gate and full proof adoption may change')
    loops[0].body[:0]=ast.parse('''_adopt_primary(runtime,verified)
update(dict(verified_original_ids=sorted(verified),pending_original_models=285-len(verified)-len(state['failed_preserved'])))
write(base/'verified_full_model_records.json',dict(records=list(verified.values()),required_original_models=285,
    full_canonical_models_verified=len(verified),all_original_experiments_complete=False,author_equivalence_certified=False))
''').body
    return ast.fix_missing_locations(tree)


def namespace():
    values=dict(original.__dict__,verify=verify,_selected_ready=selected_ready,_adopt_primary=adopt_primary)
    exec(compile(controller_ast(),__file__,'exec'),values)
    return values


def main():
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda snapshot:None)
    values=namespace()
    main=FunctionType(original.main.__code__,values)
    main()


if __name__=='__main__':main()
