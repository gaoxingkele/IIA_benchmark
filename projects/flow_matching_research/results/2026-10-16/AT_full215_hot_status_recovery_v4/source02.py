from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v2 as s


@pytest.mark.parametrize('error',[PermissionError(13,'busy'),FileNotFoundError(2,'replacing'),json.JSONDecodeError('partial','',0)])
def test_hot_status_only_is_waiting_and_a_later_read_can_succeed(tmp_path,error):
    hot=tmp_path/'status.json'
    def fail(path):raise error
    with pytest.raises(s.HotStatusNotReady):s.hot_status_read(hot,hot,fail)
    assert s.hot_status_read(hot,hot,lambda p:{'failed_preserved':{}})=={'failed_preserved':{}}


@pytest.mark.parametrize('error',[PermissionError(13,'bad_guard'),FileNotFoundError(2,'missing_proof'),json.JSONDecodeError('invalid_proof','',0)])
def test_immutable_source_data_and_proof_errors_are_never_hidden(tmp_path,error):
    def fail(path):raise error
    with pytest.raises(type(error)):s.hot_status_read(tmp_path/'guard.json',tmp_path/'status.json',fail)


def test_readiness_recovers_only_transient_hot_file_not_failed_original_guards(tmp_path,monkeypatch):
    hot=tmp_path/'producer/status.json';hot.parent.mkdir();hot.write_text('{}')
    proof=tmp_path/'native.json';proof.write_text('{}');guard=tmp_path/'guard.json';guard.write_text('{}')
    source=dict(state_root=str(hot.parent));task=dict(kind='cohort',cases=['fixture'])
    monkeypatch.setattr(s.original,'producer_proof_paths',lambda c,i:[(proof,guard)])
    current={'failure':PermissionError(13,'sharing'),'status':{},'guard':dict(passed=True,exit_code=0,resource_guard_aborted=False)}
    def reader(path):
        if Path(path)==hot:
            if current['failure']:raise current['failure']
            return current['status']
        return current['guard']
    monkeypatch.setattr(s.original,'read',reader)
    assert s.readiness({},source,task,set(),{}) is False
    current['failure']=None
    assert s.readiness({},source,task,set(),{}) is True
    current['status']={'failed_preserved':{'fixture':'actual_model_failure'}}
    with pytest.raises(ValueError):s.readiness({},source,task,set(),{})
    current['status']={};current['guard']['passed']=False
    with pytest.raises(ValueError):s.readiness({},source,task,set(),{})


def test_every_original215_task_and_all500_model_bindings_and_guards_survive():
    old=s.read(s.ROOT/s.OLD_CONFIG);new=s.registration()
    for key in ['tasks','settings','resource_lock','python','seeds','datasets','primary_endpoint','confidence_level',
        'student_t_critical_df4_975','paired_test_support','paired_calibration','macro_weighting','interval_scope']:
        assert old[key]==new[key]
    assert len(new['tasks'])==215 and len({i for t in new['tasks'] if t['kind']=='cohort' for i in t['cases']})==500
    assert len(new['recovered_initial_preworker_task_ids'])==9 and new['numerical_worker_is_frozen_v1_function']
    assert not new['numerical_attempts_repeated']
    assert new['settings']['minimum_commit_headroom_bytes']==40*2**30


def test_only_exact_hot_status_permission_errors_and_their_true_dependencies_are_eligible():
    state=s.read(s.ROOT/s.SNAPSHOT);old=s.read(s.ROOT/s.OLD_CONFIG);source=s.read(s.ROOT/s.original.PRODUCER)
    tasks={t['id']:t for t in old['tasks']}
    assert s.only_preworker_metadata_failures(state,tasks,source)
    wrong=deepcopy(state);wrong['failed_preserved']['sensitivity_layers2__smd']='Actual worker failed'
    assert not s.only_preworker_metadata_failures(wrong,tasks,source)
    wrong=deepcopy(state);wrong['failed_preserved']['main__smd']="[Errno 13] Permission denied: 'F:/immutable/checkpoint.pt'"
    assert not s.only_preworker_metadata_failures(wrong,tasks,source)
    wrong=deepcopy(state);wrong['failed_preserved']['paired__sensitivity_layers2__smd']='Preserved failed prerequisite statistics task: main__smap'
    assert not s.only_preworker_metadata_failures(wrong,tasks,source)


def test_handoff_never_touches_a_numerical_descendant_or_completed_attempt(monkeypatch):
    c=s.registration();state=s.read(s.ROOT/s.SNAPSHOT);h=c['predecessor_owner']
    monkeypatch.setattr(s,'no_original_attempts',lambda runtime:True)
    assert s.idle_allowed(state,h,[])
    assert not s.idle_allowed(state,h,[dict(name='python.exe',exe='C:/python.exe')])
    changed=deepcopy(state);changed['active_pid']=999
    assert not s.idle_allowed(changed,h,[])
    changed=deepcopy(state);changed['completed_tasks']=['main__smd']
    assert not s.idle_allowed(changed,h,[])
    monkeypatch.setattr(s,'no_original_attempts',lambda runtime:False)
    assert not s.idle_allowed(state,h,[])
