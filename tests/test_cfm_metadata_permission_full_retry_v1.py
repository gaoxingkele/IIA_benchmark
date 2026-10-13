import ast
import inspect
import json
import pytest
from scripts.flow_matching import run_cfm_metadata_safe_original_job_v1 as worker
from scripts.flow_matching import run_cfm_metadata_permission_full_retry_v1 as controller


def test_transient_replace_failure_retries_only_publication_and_preserves_exact_payload(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';path.write_text('old')
    actual=worker.os.replace;calls=[]
    def replace(source,target):
        calls.append((source,target))
        if len(calls)<3:raise PermissionError('temporary reader lock')
        return actual(source,target)
    monkeypatch.setattr(worker.os,'replace',replace);monkeypatch.setattr(worker.time,'sleep',lambda _:None)
    data={'epoch':400,'optimizer_updates':6400,'phase':'training'};worker.safe_write(path,data)
    assert path.read_text()==json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    assert len(calls)==3 and len({str(s) for s,t in calls})==1
    assert not list(tmp_path.glob('*.tmp.*'))


def test_persistent_permission_failure_is_bounded_and_preserves_previous_metadata(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';path.write_text('old');calls=[]
    def replace(*args):calls.append(args);raise PermissionError('persistent reader lock')
    monkeypatch.setattr(worker.os,'replace',replace);monkeypatch.setattr(worker.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):worker.safe_write(path,{'epoch':1})
    assert len(calls)==8 and path.read_text()=='old'
    assert not list(tmp_path.glob('*.tmp.*'))


def test_native_run_bytecode_and_all_numeric_functions_are_unchanged():
    values=worker.namespace()
    assert values['run'].__code__ is worker.original.run.__code__
    assert values['run'].__globals__ is values and values['write'] is worker.safe_write
    assert values['complete'].__code__ is worker.original.complete.__code__


def test_controller_changes_only_native_entry_and_preserves_isolated_verification():
    tree=controller.main_ast()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_cfm_metadata_safe_original_job_v1':
            node.value='scripts.flow_matching.run_cfm_ts_original_job'
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(controller.original.main)))
    values=controller.namespace()
    assert values['external_complete'].__code__ is controller.original.external_complete.__code__


def test_registered_retries_preserve_all285_slots_and_full_original_budgets():
    runtime=controller.original.read(controller.ROOT/'configs/runtime/fm_cfm_metadata_permission_full_retry.v1.json')
    chosen,queue=controller.verify_runtime(runtime)
    assert len(queue['jobs'])==285 and len(chosen)==2
    assert [j['seed'] for j in chosen]==[1104,1106]
    assert all(j['epochs']==400 and j['expected_optimizer_updates']==6400 for j in chosen)
    assert all(j['test_trajectories']==20 and j['test_points_per_trajectory']==500 for j in chosen)
