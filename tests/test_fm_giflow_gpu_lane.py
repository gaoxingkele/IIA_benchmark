import ast
import json
from pathlib import Path
import pytest


def test_native_gpu_dispatch_preserves_full_guard_and_original_child():
    from scripts.flow_matching import run_grasp_protocol_queue as original
    from scripts.flow_matching import giflow_native_protocol as protocol
    from scripts.flow_matching.run_giflow_guarded_gpu_queue_v2 import bound_main
    bound = bound_main()
    changes = {
        ('configs/experiments/fm_grasp_protocol_queue.v2.json','configs/experiments/fm_giflow_native_queue.v2.json'),
        ('scripts.flow_matching.grasp_protocol_runner','scripts.flow_matching.run_giflow_native_job'),
        ('Full GRASP GPU queue already owned','Full GiFlow GPU queue already owned')}
    assert {(a,b) for a,b in zip(original.main.__code__.co_consts,bound.__code__.co_consts) if a!=b} == changes
    node = next(n for n in ast.parse(Path(original.__file__).read_text(encoding='utf-8')).body
                if isinstance(n,ast.FunctionDef) and n.name=='main')
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),original.__file__,'exec'),namespace)
    assert bound.__code__.co_code == namespace['main'].__code__.co_code
    assert bound.__globals__['complete'] is protocol.complete
    assert bound.__globals__['verify'] is protocol.verify
    assert bound.__globals__['guarded_gpu_wait'] is original.guarded_gpu_wait


def test_registration_preserves_models_data_environment_budget_and_original_results(tmp_path):
    from scripts.flow_matching.register_giflow_guarded_gpu_lane import register
    from scripts.flow_matching.giflow_native_protocol import sha, write_json
    args = {'training_epoch':300,'batch_size':128,'stride':1,'window':24,'missing_type':'point'}
    source = {'jobs':[{'id':'slot','arguments':args,'track':'released_mirror','output_directory':'original/slot'}],
              'source_receipts':[], 'settings':{'minimum_commit_headroom_bytes':16,'emergency_commit_headroom_bytes':8,'cpu_threads':4},
              'python':'author_python','environment_lock':'lock','prediction_artifact_root':'external',
              'data_config':'data','runtime_config':'runtime','state_root':'old','requires_complete_queue':'grin'}
    write_json(tmp_path/'original.json',source)
    policy = {'original_queue':'original.json','original_queue_sha256':sha(tmp_path/'original.json'),
              'queue':'successor.json','state_root':'new','resource_lock':'grasp.lock',
              'gpu_settings':{'gpu_index':0,'gpu_uuid':'GPU-frozen','minimum_gpu_free_bytes':8,'emergency_gpu_free_bytes':2}}
    write_json(tmp_path/'policy.json',policy)
    for relative in ['scripts/flow_matching/run_grasp_protocol_queue.py',
                     'scripts/flow_matching/run_giflow_guarded_gpu_queue_v2.py',
                     'scripts/flow_matching/register_giflow_guarded_gpu_lane.py','tests/test_fm_giflow_gpu_lane.py']:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen')
    queue = register('policy.json',tmp_path)
    assert queue['jobs'] == source['jobs'] and queue['requires_complete_queue'] == 'grin'
    assert all(queue['settings'][k] == v for k,v in source['settings'].items())
    assert queue['resource_lock']=='grasp.lock' and queue['state_root']=='new'
    assert queue['cohort_count']==len(queue['jobs'])
    with pytest.raises(FileExistsError):
        register('policy.json',tmp_path)
