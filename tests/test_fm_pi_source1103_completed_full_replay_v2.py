import ast
import copy
from pathlib import Path

import pytest

from scripts.flow_matching import run_pi_source1103_completed_full_replay_v2 as pipeline


def test_entire_existing_point_native_and_full_array_review_controller_is_preserved():
    source=Path(pipeline.original.__file__).read_text(encoding='utf-8')
    expected=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    assert ast.dump(pipeline.controller_ast(source).body[0])==ast.dump(expected)


@pytest.mark.parametrize('source',['def different(): pass','def controller(): pass\ndef controller(): pass'])
def test_missing_or_duplicated_complete_controller_is_rejected(source):
    with pytest.raises(ValueError):pipeline.controller_ast(source)


def fixture(monkeypatch):
    native_template=dict(python='author',cpu_threads=2,settings=dict(commit=32,emergency=16),
        resource_lock='original_shared_lock',score_tolerance=dict(rtol=1e-6,atol=1e-7))
    review_template=dict(python='host',settings=dict(commit=22,emergency=16),resource_lock='original_shared_lock')
    point=dict(copy.deepcopy(review_template),cpu_threads=2,output_directory='point_output',
        recovery_queue='recovery',recovery_job_id=pipeline.RECOVERY)
    native=dict(copy.deepcopy(native_template),recovery_queue='recovery',recovery_job_id=pipeline.RECOVERY,
        independent_completion='point_output/independent_full_recovery_completion.json')
    job=dict(id=pipeline.CANONICAL,seed=1103,branch='source_runtime',dataset='PSM',recipe='full',
        author_config=dict(data=dict(batch_size=256,win_size=100),model=dict(d_model=512),training=dict(n_epochs=3)))
    recovered=dict(job,id=pipeline.RECOVERY,output_directory='recovery_output')
    settings=dict(seeds=[42,1103,1104,1105,1106,1107])
    bound=dict(job=recovered,original_job=job,kind='pi',original_queue='original',settings=settings)
    queue=dict(jobs=[job]+[dict(id='original_'+str(i)) for i in range(1139)],settings=settings)
    result=dict(parameter_count=1930289,active_attention_layers=1,epochs=[dict(epoch=i,steps=518,early_stop=False) for i in [1,2,3]],
        native_train_points=132481,training_windows=132382,native_test_points=87841,scored_test_points=87800,dropped_test_tail_points=41)
    config=dict(source_receipts=[],original_queue='original',recovery_queue='recovery',native_template='native_template',
        review_template='review_template',required_original_jobs=1140,original_scope=copy.deepcopy(pipeline.SCOPE),
        additional_training_seed_slots=0,required_original_training_updates=1554,required_full_train_score_points=13238200,
        required_full_test_score_points=87800,required_full_saved_score_streams=15,
        cases=[dict(canonical_id=pipeline.CANONICAL,point_configuration='point',native_configuration='native')])
    mapping=dict(original=queue,recovery=dict(jobs=[bound]),native_template=native_template,review_template=review_template,
        point=point,native=native)
    mapping['recovery_output/resource_usage.json']=dict(resource_guard_aborted=False)
    monkeypatch.setattr(pipeline,'read',lambda p:mapping[p])
    monkeypatch.setattr(pipeline.original.completion,'binding',lambda p:(bound,job,result))
    return config,mapping,bound,job,result


def test_exact_full_scope_original_recovery_and_all_three_stage_guards_accepted(monkeypatch):
    config,mapping,_,_,_=fixture(monkeypatch)
    queue,native,review=pipeline.verify(config)
    assert len(queue['jobs'])==1140 and native==mapping['native_template'] and review==mapping['review_template']


@pytest.mark.parametrize('field,value',[
    ('required_original_jobs',1),('additional_training_seed_slots',1),('required_original_training_updates',1553),
    ('required_full_train_score_points',13238199),('required_full_test_score_points',87799),
    ('required_full_saved_score_streams',14),('original_scope',dict(papers=44,method_baseline_ablation_records=681,original_data_task_records=273)),
])
def test_smaller_scope_partial_full_scores_or_additional_seeds_rejected(monkeypatch,field,value):
    config,_,_,_,_=fixture(monkeypatch);config[field]=value
    with pytest.raises(ValueError):pipeline.verify(config)


@pytest.mark.parametrize('change',['missing_original','duplicate_original','seed_list','wrong_case','native_guard','point_guard',
    'native_threads','point_threads','native_lock','point_lock','tolerance','point_recovery','native_recovery','completion_binding'])
def test_original_scope_identity_settings_and_frozen_tolerance_cannot_change(monkeypatch,change):
    config,mapping,_,_,_=fixture(monkeypatch)
    if change=='missing_original':mapping['original']['jobs'].pop()
    elif change=='duplicate_original':mapping['original']['jobs'][-1]=copy.deepcopy(mapping['original']['jobs'][0])
    elif change=='seed_list':mapping['original']['settings']['seeds']=[1105]
    elif change=='wrong_case':config['cases'][0]['canonical_id']='different'
    elif change=='native_guard':mapping['native']['settings']['commit']=16
    elif change=='point_guard':mapping['point']['settings']['commit']=16
    elif change=='native_threads':mapping['native']['cpu_threads']=1
    elif change=='point_threads':mapping['point']['cpu_threads']=1
    elif change=='native_lock':mapping['native']['resource_lock']='different'
    elif change=='point_lock':mapping['point']['resource_lock']='different'
    elif change=='tolerance':mapping['native']['score_tolerance']['atol']=.1
    elif change=='point_recovery':mapping['point']['recovery_job_id']='different'
    elif change=='native_recovery':mapping['native']['recovery_queue']='different'
    elif change=='completion_binding':mapping['native']['independent_completion']='different'
    with pytest.raises(ValueError):pipeline.verify(config)


@pytest.mark.parametrize('change',['seed','branch','dataset','recipe','kind','parameters','layers','epochs','steps','early_stop',
    'train_points','train_windows','test_points','scored_test','tail','batch','window','width','budget','aborted','unknown_guard'])
def test_full_original_native_model_and_training_completion_required(monkeypatch,change):
    config,mapping,bound,job,result=fixture(monkeypatch)
    if change=='seed':job['seed']=1104
    elif change=='branch':job['branch']='causal_axis_repaired'
    elif change=='dataset':job['dataset']='SMD'
    elif change=='recipe':job['recipe']='ablation'
    elif change=='kind':bound['kind']='readonly_original_canonical_binding'
    elif change=='parameters':result['parameter_count']=1930288
    elif change=='layers':result['active_attention_layers']=2
    elif change=='epochs':result['epochs'].pop()
    elif change=='steps':result['epochs'][0]['steps']=517
    elif change=='early_stop':result['epochs'][-1]['early_stop']=True
    elif change=='train_points':result['native_train_points']-=1
    elif change=='train_windows':result['training_windows']-=1
    elif change=='test_points':result['native_test_points']-=1
    elif change=='scored_test':result['scored_test_points']-=1
    elif change=='tail':result['dropped_test_tail_points']=0
    elif change=='batch':job['author_config']['data']['batch_size']=64
    elif change=='window':job['author_config']['data']['win_size']=50
    elif change=='width':job['author_config']['model']['d_model']=128
    elif change=='budget':job['author_config']['training']['n_epochs']=2
    elif change=='aborted':mapping['recovery_output/resource_usage.json']['resource_guard_aborted']=True
    elif change=='unknown_guard':mapping['recovery_output/resource_usage.json'].clear()
    with pytest.raises(ValueError):pipeline.verify(config)


def test_changed_frozen_source_is_rejected_before_any_stage_launch(monkeypatch):
    config,_,_,_,_=fixture(monkeypatch)
    config['source_receipts']=[dict(path='original_model',sha256='before')]
    monkeypatch.setattr(pipeline,'digest',lambda _:'changed')
    with pytest.raises(ValueError,match='Frozen'):pipeline.verify(config)


def test_all_original1105_full_stage_checks_remain_except_new_completed_seed_identity():
    import inspect
    from scripts.flow_matching import run_pi_source1105_completed_full_replay_v1 as prior
    actual=ast.parse(inspect.getsource(pipeline.verify));changed=0
    for node in ast.walk(actual):
        if isinstance(node,ast.Compare) and ast.unparse(node.left)=="job['seed']" and node.comparators[0].value==1103:
            node.comparators[0].value=1105;changed+=1
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and 'PSM1103' in node.value:
            node.value=node.value.replace('PSM1103','PSM1105')
    assert changed==1 and ast.dump(actual)==ast.dump(ast.parse(inspect.getsource(prior.verify)))
