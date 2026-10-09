import ast
import json
from pathlib import Path
import pytest
from scripts.flow_matching import run_mtsbench_stat_queue as original
from scripts.flow_matching.run_mtsbench_stat_queue_v2 import bound_main


def test_only_configured_mutex_binding_changes_preserved_dispatch(monkeypatch,tmp_path):
    controller=bound_main()
    root=Path(original.__file__).resolve().parents[2]
    tree=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(original.__file__),'exec'),namespace)
    assert controller.__code__.co_names==namespace['main'].__code__.co_names
    old=[c for c in namespace['main'].__code__.co_consts if isinstance(c,str)]
    new=[c for c in controller.__code__.co_consts if isinstance(c,str)]
    assert [(a,b) for a,b in zip(old,new) if a!=b]==[('experiments/runs/fm_heavy_recovery_cpu.lock','resource_lock')]
    policy=json.loads((root/'configs/runtime/fm_mtsbench_stat_lane.v2.json').read_text(encoding='utf-8'))
    assert policy['resource_lock']=='experiments/runs/fm_mtsbench_stat_cpu.lock'
    assert policy['state_root']!='experiments/runs/fm_grasp_stat_native_v1'


def test_new_registration_preserves_jobs_artifacts_environment_and_all_guards(tmp_path):
    from scripts.flow_matching.register_mtsbench_stat_lane import register
    from scripts.flow_matching.mtsbench_stat_protocol import sha,write_json
    source={'source_receipts':[],'jobs':[{'id':'a','protocol':'native_transductive_test_feature_fit_oracle_f1'}],
            'settings':{'minimum_commit_headroom_bytes':16,'emergency_commit_headroom_bytes':8,'cpu_threads':1},
            'artifact_root':'external','environment_lock':'environment.json','state_root':'old'}
    write_json(tmp_path/'original.json',source)
    policy={'original_queue':'original.json','original_queue_sha256':sha(tmp_path/'original.json'),
            'queue':'new.json','state_root':'new','resource_lock':'new.lock'}
    write_json(tmp_path/'policy.json',policy)
    for path in ['scripts/flow_matching/run_mtsbench_stat_queue_v2.py','scripts/flow_matching/register_mtsbench_stat_lane.py',
                 'tests/test_fm_mtsbench_stat_lane.py']:
        file=tmp_path/path
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text('source',encoding='utf-8')
    queue=register('policy.json',tmp_path)
    for key in ['jobs','settings','artifact_root','environment_lock']:
        assert queue[key]==source[key]
    assert queue['state_root']=='new'
    with pytest.raises(FileExistsError):
        register('policy.json',tmp_path)
