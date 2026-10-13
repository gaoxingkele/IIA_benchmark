"""Cold handoff of idle full285 replay control; all numerical workers unchanged.

Every source/artifact byte remains SHA256 checked using bounded reads. The
actual128 guarded native+independent certificates survive. Parent imports no
NumPy/SciPy/Torch; fingerprint is the unchanged original pure metadata function.
CFM-TS: OpenReview Hqn4Aj7xrQ.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType

from scripts.flow_matching import run_cfm_completion_replay_queue_v1 as original
from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as handoff

ROOT=original.ROOT


def streaming_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):digest.update(block)
    return digest.hexdigest()


def validate_closed_primary_record(runtime,record):
    identity=record['original_canonical_id']
    route=next(r for r in runtime['routes'] if r['original_canonical_id']==identity)
    if record['result_path']!=route['result_path'] or record['execution_id']!=route['job']['id']:
        raise ValueError('Exact original canonical execution required')
    configpath=ROOT/runtime['child_config_root']/(identity+'.native.json')
    config=original.read(configpath)
    reportpath=ROOT/config['output_directory']/'checkpoint_replay_audit.json'
    receiptpath=ROOT/config['state_root']/'receipt.json'
    report,receipt=original.read(reportpath),original.read(receiptpath)
    original.check_native_terminal(config,report,receipt)
    reviewpath=ROOT/runtime['child_config_root']/(identity+'.review.json')
    review=original.read(reviewpath);proof=original.read(ROOT/review['output_proof'])
    guard=original.read(ROOT/review['state_root']/'receipt.json')
    if (not proof['passed'] or proof['records']!=[record] or guard['exit_code']!=0
        or guard['resource_guard_aborted'] or not guard['independent_full_array_review_passed']
        or proof['configuration_sha256']!=streaming_sha(reviewpath)
        or proof['native_report_sha256']!=streaming_sha(reportpath)
        or proof['native_receipt_sha256']!=streaming_sha(receiptpath)
        or report['configuration_sha256']!=streaming_sha(configpath)):
        raise ValueError('Real unchanged closed native/independent guarded proof chain required')
    for path,expected in [(ROOT/record['result_path'],record['result_sha256']),
        (record['checkpoint_path'],record['checkpoint_sha256']),
        (record['replay_arrays'],record['replay_arrays_sha256']),
        (ROOT/route['job']['data_path'],record['original_full_data_sha256'])]:
        if streaming_sha(path)!=expected:raise ValueError('Every previously verified artifact byte must remain')
    result=original.read(ROOT/record['result_path'])
    for artifact in result['artifacts']:
        if streaming_sha(artifact['path'])!=artifact['sha256']:raise ValueError('Original source predictions/checkpoint changed')


def native_registration_ast():
    tree=ast.parse(inspect.getsource(original.native_registration));changed=0
    for index,node in enumerate(tree.body[0].body):
        if isinstance(node,ast.ImportFrom) and node.module=='scripts.flow_matching.run_cfm_ts_original_job':
            if [n.name for n in node.names]!=['fingerprint']:raise ValueError('Only metadata fingerprint import may change')
            tree.body[0].body[index]=ast.parse('fingerprint = _fingerprint').body[0];changed+=1
    if changed!=1:raise ValueError('Exactly one lazy pure metadata fingerprint binding required')
    return ast.fix_missing_locations(tree)


def cached_api():
    api=original.resource_api()
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api['resource_snapshot']=resource_snapshot
    return api


def namespace(config):
    values=dict(original.__dict__,sha=streaming_sha,resource_api=cached_api,hashlib=hashlib)
    from scripts.flow_matching.run_light_controller import load_functions
    load_functions(ROOT,'scripts/flow_matching/run_cfm_ts_original_job.py',['fingerprint'],values)
    values['_fingerprint']=values['fingerprint']
    for name,value in original.__dict__.items():
        if isinstance(value,FunctionType) and value.__module__==original.__name__:
            rebound=FunctionType(value.__code__,values,value.__name__,value.__defaults__,value.__closure__)
            rebound.__kwdefaults__=value.__kwdefaults__;values[name]=rebound
    baseverify=values['verify']
    def verify(runtime):
        routes,template,records=baseverify(runtime)
        snapshot=original.read(ROOT/config['preserved_verified_records'])
        for record in snapshot['records']:
            identity=record['original_canonical_id']
            if identity in records:
                if records[identity]!=record:raise ValueError('Existing original85 certificate changed')
            else:
                validate_closed_primary_record(runtime,record);records[identity]=record
        if len(records)!=config['expected_verified_models']:raise ValueError('Every actual128 passed certificate required')
        return routes,template,records
    values['verify']=verify
    exec(compile(native_registration_ast(),__file__,'exec'),values)
    return values


def verify_config(config):
    for pin in config['source_receipts']:
        if streaming_sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen idle handoff changed: '+pin['path'])
    runtime=original.read(ROOT/config['original_runtime'])
    if config['expected_verified_models']!=128 or runtime['original_scope']!=config['original_scope']:
        raise ValueError('Full original285/128 state and research scope required')
    values=namespace(config);routes,_,records=values['verify'](runtime)
    if len(routes)!=285:raise ValueError('Every285 native route remains')
    return runtime,values,records


def idle_allowed(role,state,output_exists,descendants):
    return (role=='original' and state.get('active_pid') is None
        and state.get('state')=='waiting_actual_original_training_completion'
        and len(state.get('verified_original_ids',[]))==128 and not state.get('failed_preserved')
        and all(handoff.allowed_console_child(p['name'],p['exe']) for p in descendants))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=original.read(ROOT/args.config)
    runtime,values,records=verify_config(config)
    if args.probe_only:
        import psutil
        print(dict(original_models=285,actual_guarded_certificates=len(records),
            unchanged_controller_bytecode=values['controller'].__code__ is original.controller.__code__,
            every_artifact_byte_hashed=True,private_bytes=psutil.Process().memory_info().private,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules));return
    base=ROOT/config['handoff_state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=cached_api()['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Cold handoff already owned')
    try:
        function=FunctionType(handoff.safe_handoff.__code__,dict(handoff.__dict__,handoff_phase_allowed=idle_allowed))
        route=dict(previous_owner=config['previous_owner'],previous_state_path=config['previous_state_path'],runtime_configuration=args.config)
        function('original',route,base,lambda status:original.write(base/'status.json',status),10)
        verify_config(config)
        values['controller'](runtime,config['original_runtime'])
    finally:owner.close()


if __name__=='__main__':main()
