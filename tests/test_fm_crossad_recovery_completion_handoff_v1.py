"""Reject mismatched recovery science, retry seed inflation and false completion."""
import copy

import pytest

from scripts.flow_matching import run_crossad_recovery_completion_handoff_v1 as handoff


def fixture():
    job=dict(id='canonical',output_directory='failed',seed=2025,dataset='PSM',mode='fresh_train',
        train_parameters=dict(train_epochs=20,patience=3,batch_size=32),model_parameters=dict(seq_len=100),
        runner='original_runner.py',frozen_files=[dict(path='source.py',sha256='frozen')])
    recovery=copy.deepcopy(job);recovery.update(id='retry',output_directory='new_attempt')
    parent=dict(settings=dict(raw_root='raw',cpu_threads=2,output_root='original',evaluation=dict(points=[.1,1,5])),jobs=[job])
    queue=copy.deepcopy(parent);queue['jobs']=[recovery]
    route=dict(canonical_id='canonical',recovery_job_id='retry',kind='registered_exact_recovery')
    return route,job,queue,parent


def test_exact_route_preserves_science():
    route,job,queue,parent=fixture()
    assert handoff.exact_route(route,job,queue,parent)==queue['jobs'][0]
    assert job['output_directory']=='failed'


@pytest.mark.parametrize('key,value',[
    ('seed',1103),('dataset','SMAP'),('mode','release_checkpoint'),('runner','alternate.py'),
    ('model_parameters',dict(seq_len=64)),('train_parameters',dict(train_epochs=1,patience=3,batch_size=1)),
    ('frozen_files',[]),('output_directory','failed'),
])
def test_changed_science_or_failed_output_is_rejected(key,value):
    route,job,queue,parent=fixture();queue['jobs'][0][key]=value
    with pytest.raises(ValueError):handoff.exact_route(route,job,queue,parent)


@pytest.mark.parametrize('key,value',[
    ('raw_root','different_data'),('cpu_threads',1),('output_root','unregistered'),('evaluation',dict(points=[1])),
])
def test_settings_changes_are_rejected(key,value):
    route,job,queue,parent=fixture();queue['settings'][key]=value
    with pytest.raises(ValueError):handoff.exact_route(route,job,queue,parent)


def test_selected_full_retry_can_keep_id_but_only_change_registered_output_metadata():
    route,job,queue,parent=fixture()
    route.update(kind='selected_full_retry',recovery_job_id='canonical',selected_state_root='new_controller')
    queue['jobs'][0]['id']='canonical';queue['settings']['output_root']='new_controller'
    assert handoff.exact_route(route,job,queue,parent)['seed']==2025
    queue['settings']['cpu_threads']=1
    with pytest.raises(ValueError):handoff.exact_route(route,job,queue,parent)


def test_ambiguous_or_missing_recovery_id_rejected():
    route,job,queue,parent=fixture();queue['jobs']*=2
    with pytest.raises(ValueError):handoff.exact_route(route,job,queue,parent)
    queue['jobs']=[]
    with pytest.raises(ValueError):handoff.exact_route(route,job,queue,parent)


def test_canonical_union_does_not_count_retry_as_an_additional_seed():
    ids=list(handoff.EXPECTED)+['original_'+str(i) for i in range(135)]
    result=handoff.canonical_union(ids,{ids[0]:{},ids[1]:{}},handoff.EXPECTED)
    assert result['verified_original_jobs']==2 and result['pending_original_jobs']==137
    assert result['additional_training_seed_slots']==0
    assert not result['all_original_experiments_complete']
    with pytest.raises(ValueError):handoff.canonical_union(ids,{'retry_id':{}},handoff.EXPECTED)


def test_full_family_completion_does_not_certify_whole45_paper_objective():
    ids=list(handoff.EXPECTED)+['original_'+str(i) for i in range(135)]
    result=handoff.canonical_union(ids,dict.fromkeys(ids),handoff.EXPECTED)
    assert result['all139_original_full_native_and_independent_executions_verified']
    assert result['pending_original_jobs']==0
    assert result['all_original_experiments_complete'] is False


@pytest.mark.parametrize('bad_ids,bad_routes',[(138,4),(140,4),(139,3),(139,5)])
def test_narrowed_or_inflated_scope_rejected(bad_ids,bad_routes):
    ids=list(handoff.EXPECTED)+['original_'+str(i) for i in range(bad_ids-4)]
    routes=list(handoff.EXPECTED)[:bad_routes]
    if bad_routes==5:routes+=['extra_retry']
    with pytest.raises(ValueError):handoff.canonical_union(ids,{},routes)


def test_failed_stage_cannot_be_silently_retried(monkeypatch,tmp_path):
    monkeypatch.setattr(handoff,'ROOT',tmp_path)
    (tmp_path/'child').mkdir();(tmp_path/'child/receipt.json').write_text('{}')
    monkeypatch.setattr(handoff,'read',lambda p:dict(state_root='child') if p=='config' else
        dict(exit_code=7,resource_guard_aborted=False))
    with pytest.raises(ValueError,match='no silent retry'):
        handoff.run_stage({},'config','anything','native','canonical',lambda _:None)


def test_resource_aborted_zero_exit_cannot_be_reused(monkeypatch,tmp_path):
    monkeypatch.setattr(handoff,'ROOT',tmp_path)
    (tmp_path/'child').mkdir();(tmp_path/'child/receipt.json').write_text('{}')
    monkeypatch.setattr(handoff,'read',lambda p:dict(state_root='child') if p=='config' else
        dict(exit_code=0,resource_guard_aborted=True))
    with pytest.raises(ValueError):
        handoff.run_stage({},'config','anything','native','canonical',lambda _:None)


def review_fixture(monkeypatch):
    guards=dict(python='author_python',cpu_threads=2,settings=dict(emergency_commit_headroom_bytes=16),resource_lock='original_lock')
    job=dict(id='retry',seed=2025,mode='fresh_train',output_directory='original_result')
    native=dict(guards,queue='native_queue',job_id='retry',state_root='native_state',
        artifact_directory='native_arrays',independent_point_completion='point')
    post=dict(guards,native_configuration='native_config',output_directory='post',state_root='post_state',source_receipts=[])
    report=dict(configuration_sha256='hash',full_score_points=dict(train=10,validation=2,test=8))
    receipt=dict(exit_code=0,resource_guard_aborted=False,independent_stage_passed=True)
    proof=dict(passed=True,configuration_sha256='hash',job_id='retry',original_seed=2025,mode='fresh_train',
        full_point_counts=report['full_score_points'],original_training_seed_slots_completed=1,
        required_original_cohort_training_seeds=6,additional_training_seed_slots=0,strict_TAB_result=False,
        author_equivalence_certified=False,all_original_experiments_complete=False,native_proof_sha256='hash',
        full_native_model_replay_passed=True,all_original_raw_point_and_row_identities_independently_reviewed=True,
        all_full_original_and_replayed_scores_independently_compared=True,
        all_original_author_metrics_reexecuted_and_compared=True,
        replayed_validation_only_metrics_independently_recomputed={'0.1':{},'1':{},'5':{}})
    point=dict(passed=True,job_id='retry',full_raw_labels_and_source_rows_checked=True,original_source_result={'result':'full'})
    mapping={'post_config':post,'native_config':native,'native_queue':{'jobs':[job]},'point':point,
        'post/independent_completed_replay_review.json':proof,'post_state/receipt.json':receipt}
    monkeypatch.setattr(handoff,'read',lambda p:mapping.get(str(p),{'result':'full'}))
    monkeypatch.setattr(handoff,'digest',lambda p:'hash')
    monkeypatch.setattr(handoff.original,'native_terminal',lambda config,j:(report,dict(exit_code=0,resource_guard_aborted=False)))
    return guards,job,proof,receipt,point,report,post


def test_complete_native_and_independent_recovery_proof_accepted(monkeypatch):
    guards,job,*_=review_fixture(monkeypatch)
    result=handoff.completed_review('post_config',job,guards)
    assert result['completed_original_training_seed_slots']==1


@pytest.mark.parametrize('target,key,value',[
    ('proof','passed',False),('proof','original_seed',1103),('proof','job_id','another_retry'),
    ('proof','strict_TAB_result',True),('proof','author_equivalence_certified',True),
    ('proof','all_original_experiments_complete',True),('proof','additional_training_seed_slots',1),
    ('proof','required_original_cohort_training_seeds',1),('proof','original_training_seed_slots_completed',2),
    ('proof','full_point_counts',dict(test=1)),('proof','configuration_sha256','changed'),
    ('proof','replayed_validation_only_metrics_independently_recomputed',{'1':{}}),
    ('proof','all_full_original_and_replayed_scores_independently_compared',False),
    ('proof','all_original_author_metrics_reexecuted_and_compared',False),
    ('receipt','exit_code',1),('receipt','resource_guard_aborted',True),('receipt','independent_stage_passed',False),
    ('point','full_raw_labels_and_source_rows_checked',False),('point','original_source_result',{'partial':True}),
    ('report','configuration_sha256','changed'),('post','cpu_threads',1),
])
def test_incomplete_or_false_review_cannot_supply_canonical_credit(monkeypatch,target,key,value):
    guards,job,proof,receipt,point,report,post=review_fixture(monkeypatch)
    objects=dict(proof=proof,receipt=receipt,point=point,report=report,post=post)
    objects[target][key]=value
    with pytest.raises(ValueError):handoff.completed_review('post_config',job,guards)
