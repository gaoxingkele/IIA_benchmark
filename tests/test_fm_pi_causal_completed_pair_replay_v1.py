import ast
import copy
from pathlib import Path

import pytest

from scripts.flow_matching import run_pi_causal_completed_pair_replay_v1 as pipeline


def test_complete_audit_only_changes_disclosure_of_absent_historical_resource_receipt():
    source=Path(pipeline.completion.__file__).read_text(encoding='utf-8')
    tree=ast.parse(source);audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    patched=pipeline.completion_ast(source).body[0]
    expected=copy.deepcopy(audit)
    assignment=next(n for n in expected.body if isinstance(n,ast.Assign) and
        isinstance(n.targets[0],ast.Name) and n.targets[0].id=='usage')
    assignment.value=ast.parse('historical_training_usage(base,job,case)',mode='eval').body
    assert ast.dump(patched)==ast.dump(expected)


@pytest.mark.parametrize('source',[
    'def audit(config, path):\n usage = {}\n',
    "def audit(config, path):\n usage = read(base/'resource_usage.json')\n usage = read(base/'resource_usage.json')\n",
])
def test_moved_or_duplicated_frozen_audit_boundary_is_rejected(source):
    with pytest.raises(ValueError):pipeline.completion_ast(source)


def test_missing_canonical_historical_guard_is_explicitly_unproven(tmp_path):
    job=dict(id='original',output_directory='canonical')
    case=dict(job=copy.deepcopy(job),kind='readonly_original_canonical_binding')
    result=pipeline.historical_training_usage(tmp_path,job,case)
    assert result['resource_guard_aborted'] is None
    assert result['original_training_resource_evidence_available'] is False
    assert result['original_training_guard_verified'] is False


@pytest.mark.parametrize('change',['kind','job'])
def test_missing_recovery_guard_cannot_be_hidden_as_historical_canonical_output(tmp_path,change):
    job=dict(id='original',output_directory='canonical')
    case=dict(job=copy.deepcopy(job),kind='readonly_original_canonical_binding')
    if change=='kind':case['kind']='recovery'
    else:case['job']['output_directory']='retry'
    with pytest.raises(ValueError):pipeline.historical_training_usage(tmp_path,job,case)


@pytest.mark.parametrize('flag',[True,None])
def test_existing_aborted_or_unknown_training_receipt_is_rejected(tmp_path,monkeypatch,flag):
    (tmp_path/'resource_usage.json').write_text('{}')
    monkeypatch.setattr(pipeline,'read',lambda _:dict(resource_guard_aborted=flag))
    with pytest.raises(ValueError):pipeline.historical_training_usage(tmp_path,{}, {})


def test_actual_nonaborted_original_receipt_is_preserved(tmp_path,monkeypatch):
    (tmp_path/'resource_usage.json').write_text('{}')
    monkeypatch.setattr(pipeline,'read',lambda _:dict(resource_guard_aborted=False,peak_private_bytes=123))
    result=pipeline.historical_training_usage(tmp_path,{}, {})
    assert result['original_training_resource_evidence_available'] and result['peak_private_bytes']==123


@pytest.mark.parametrize('receipt',[
    dict(exit_code=1,resource_guard_aborted=False,passed=True),
    dict(exit_code=0,resource_guard_aborted=True,passed=True),
    dict(exit_code=0,resource_guard_aborted=False,passed=False),
])
def test_false_or_aborted_full_stage_does_not_advance(monkeypatch,receipt):
    monkeypatch.setattr(pipeline,'read',lambda _:receipt)
    with pytest.raises(ValueError):pipeline.successful_stage(dict(state_root='state'),'passed')


def test_actual_successful_nonaborted_stage_can_advance(monkeypatch):
    receipt=dict(exit_code=0,resource_guard_aborted=False,passed=True)
    monkeypatch.setattr(pipeline,'read',lambda _:receipt)
    assert pipeline.successful_stage(dict(state_root='state'),'passed')==receipt


def verify_fixture(monkeypatch):
    native=dict(python='author',cpu_threads=2,settings=dict(minimum_commit_headroom_bytes=32),resource_lock='lock',
        score_tolerance=dict(rtol=1e-6,atol=1e-7))
    review=dict(python='host',settings=dict(minimum_commit_headroom_bytes=22),resource_lock='lock')
    queue=dict(jobs=[],settings=dict(seeds=[42,1103,1104,1105,1106,1107]))
    config=dict(original_queue='original',native_template='native_template',review_template='review_template',
        source_receipts=[],required_original_jobs=1140,cases=[],original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273))
    mapping=dict(original=queue,native_template=native,review_template=review);bindings={}
    for i,identity in enumerate(pipeline.IDS):
        job=dict(id=identity,seed=1103+i,branch='causal_axis_repaired',dataset='PSM',recipe='full',
            author_config=dict(data=dict(batch_size=256),model=dict(d_model=512),training=dict(n_epochs=3)))
        queue['jobs'].append(job)
        point=dict(review,cpu_threads=2,output_directory='point'+str(i),recovery_queue='readonly')
        replay=dict(native,recovery_job_id=identity,recovery_queue='readonly',
            independent_completion=point['output_directory']+'/independent_full_recovery_completion.json')
        pp='point'+str(i);np='native'+str(i);mapping[pp]=point;mapping[np]=replay
        config['cases'].append(dict(canonical_id=identity,point_configuration=pp,native_configuration=np))
        bindings[id(point)]=(dict(job=job,kind='readonly_original_canonical_binding',settings=queue['settings']),job,
            dict(parameter_count=1930289,epochs=[dict(steps=518)]*3))
    queue['jobs'] += [dict(id='original_'+str(i)) for i in range(1138)]
    monkeypatch.setattr(pipeline,'read',lambda p:mapping[p])
    monkeypatch.setattr(pipeline.completion,'binding',lambda p:bindings[id(p)])
    return config,mapping,bindings


def test_full1140_scope_and_original_native_independent_guards_accepted(monkeypatch):
    config,mapping,_=verify_fixture(monkeypatch)
    assert len(pipeline.verify(config)[0]['jobs'])==1140


@pytest.mark.parametrize('change',['scope','original_count','missing_job','missing_case','seed_list',
    'native_guard','point_guard','cpu_threads','score_tolerance','wrong_branch','wrong_seed','short_epochs','steps','width','batch'])
def test_scope_budget_or_resource_changes_are_rejected(monkeypatch,change):
    config,mapping,bindings=verify_fixture(monkeypatch)
    job=mapping['original']['jobs'][0];point=mapping['point0'];native=mapping['native0']
    if change=='scope':config['original_scope']['papers']=44
    elif change=='original_count':config['required_original_jobs']=2
    elif change=='missing_job':mapping['original']['jobs'].pop()
    elif change=='missing_case':config['cases'].pop()
    elif change=='seed_list':mapping['original']['settings']['seeds']=[1103,1104]
    elif change=='native_guard':native['settings']=dict(minimum_commit_headroom_bytes=16)
    elif change=='point_guard':point['settings']=dict(minimum_commit_headroom_bytes=16)
    elif change=='cpu_threads':native['cpu_threads']=1
    elif change=='score_tolerance':native['score_tolerance']=dict(rtol=.1,atol=.1)
    elif change=='wrong_branch':job['branch']='source_runtime'
    elif change=='wrong_seed':job['seed']=1112
    elif change=='short_epochs':bindings[id(point)][2]['epochs'].pop()
    elif change=='steps':bindings[id(point)][2]['epochs']=[dict(steps=1)]*3
    elif change=='width':job['author_config']['model']['d_model']=128
    elif change=='batch':job['author_config']['data']['batch_size']=64
    with pytest.raises(ValueError):pipeline.verify(config)
