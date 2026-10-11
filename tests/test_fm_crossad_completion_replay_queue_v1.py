import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as queue
from scripts.flow_matching.replay_crossad_registered_model_v3 import select_factory


@pytest.mark.parametrize('row',[None,1,2,3,4,5])
def test_replay_constructs_exact_frozen_full_or_reconstructed_ablation_model(row):
    calls=[]
    full=lambda runtime,parameters: calls.append(('full',runtime,parameters)) or 'full'
    ablation=lambda runtime,parameters,number: calls.append(('ablation',runtime,parameters,number)) or 'ablation'
    checkpoint=lambda model: calls.append(('pure_checkpoint',model)) or model
    job=dict(mode='fresh_train',reconstructed_ablation=row is not None)
    if row is not None:
        job.update(ablation_row=row,author_recipe='paper_reconstructed_table3_row'+str(row),
            runner='scripts/flow_matching/run_crossad_ablation_v2_job.py')
    parameters={'seq_len':128,'bank_size':32}
    factory=select_factory(job,full,ablation,checkpoint)
    factory('frozen-runtime',parameters)
    assert calls[0][0]==('ablation' if row is not None else 'full')
    assert calls[0][2] is parameters
    if row is not None:
        assert calls[0][3]==row and calls[1]==('pure_checkpoint','ablation')


@pytest.mark.parametrize('field,value',[('ablation_row',6),('mode','release_checkpoint'),
    ('reconstructed_ablation',False),('author_recipe','fresh_train'),('runner','other.py')])
def test_ablation_cannot_silently_become_full_author_or_released_model(field,value):
    job=dict(mode='fresh_train',reconstructed_ablation=True,ablation_row=3,
        author_recipe='paper_reconstructed_table3_row3',runner='scripts/flow_matching/run_crossad_ablation_v2_job.py')
    job[field]=value
    with pytest.raises(ValueError): select_factory(job,None,None,None)


@pytest.mark.parametrize('failure',[None,'missing','duplicate','seed','ablation','scope','operating_points'])
def test_preserve_every_original_139_execution_and_six_seed_training_obligations(failure):
    path=Path(__file__).resolve().parents[1]/'configs/experiments/fm_crossad_complete_queue.v7.json'
    source=json.loads(path.read_text(encoding='utf-8'))
    runtime=dict(required_original_jobs=139,original_scope=copy.deepcopy(queue.SCOPE))
    if failure=='missing':source['jobs'].pop()
    elif failure=='duplicate':source['jobs'][-1]=source['jobs'][0]
    elif failure=='seed':source['jobs'][-1]['seed']=2025
    elif failure=='ablation':source['jobs'][-1].pop('ablation_row')
    elif failure=='scope':runtime['original_scope']['papers']=44
    elif failure=='operating_points':source['settings']['evaluation']['strict_false_alarm_percent']=[1]
    if failure:
        with pytest.raises(ValueError):queue.validate_scope(source,runtime)
    else:assert len(queue.validate_scope(source,runtime))==139


@pytest.mark.parametrize('field,value',[('status','partial'),('seed',1104),('mode','release_checkpoint'),
    ('dataset','other'),('experiment_sha256','changed'),('paper_equivalence',True),('strict_TAB_result',True)])
def test_partial_or_scientifically_different_source_execution_cannot_enter_replay(field,value):
    job=dict(id='original',seed=1103,mode='fresh_train',dataset='PSM',train_parameters={'train_epochs':20,'patience':3})
    result=dict(status='completed',id=job['id'],seed=job['seed'],mode=job['mode'],dataset=job['dataset'],
        experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
        paper_equivalence=False,strict_TAB_result=False,details=dict(declared_train_parameters=job['train_parameters']))
    queue.complete_result(job,result)
    result[field]=value
    with pytest.raises(ValueError):queue.complete_result(job,result)


def test_existing_stage_configuration_is_immutable(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    queue.exclusive_configuration('original.json',dict(seed=1103,full_budget=20))
    queue.exclusive_configuration('original.json',dict(seed=1103,full_budget=20))
    with pytest.raises(ValueError):queue.exclusive_configuration('original.json',dict(seed=1103,full_budget=1))
    assert json.loads((tmp_path/'original.json').read_text())['full_budget']==20


def test_new_controller_parent_keeps_heavy_numeric_imports_inside_guarded_workers():
    import ast
    tree=ast.parse(Path(queue.__file__).read_text(encoding='utf-8'))
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('numpy' not in ast.unparse(n) and 'torch' not in ast.unparse(n) for n in imports)
    controller=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller'))
    assert 'replay_crossad_registered_model_v3' in controller and 'live_stage_jobs()' in controller
    assert 'run_crossad_author_job' not in controller


@pytest.mark.parametrize('failure',[None,'resource_abort','nonzero_exit','short_points','dropped_weights',
    'missing_spot','changed_range_grid','looser_scores','looser_metrics','added_seed','release_as_training'])
def test_native_terminal_rejects_partial_or_aborted_model_and_changed_original_evaluation(tmp_path,monkeypatch,failure):
    job=dict(id='original',seed=2025,mode='release_checkpoint',dataset='MSL',
        train_parameters={'train_epochs':20,'patience':3},output_directory='original')
    result=dict(status='completed',id=job['id'],seed=job['seed'],mode=job['mode'],dataset=job['dataset'],
        experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
        paper_equivalence=False,strict_TAB_result=False,details=dict(declared_train_parameters=job['train_parameters'],
            native_points={'train':2,'validation':2,'test':2},parameter_count=10))
    target=tmp_path/'original/result.json';target.parent.mkdir();target.write_text(json.dumps(result),encoding='utf-8')
    config=dict(artifact_directory='native',state_root='state',cpu_threads=2)
    report=dict(job_id=job['id'],original_seed=job['seed'],source_result_sha256=queue.digest(target),
        full_original_model_reinference_passed=True,full_original_author_metric_reexecution_passed=True,
        original_SPOT_init_model_inference_and_threshold_reexecution_passed=True,
        original_full_state_dictionary_loaded_strictly=True,cuda_context_initialized=False,
        additional_training_seed_slots=0,full_score_points=result['details']['native_points'],full_model_parameters=10,
        actual_cpu_threads=2,original_inference_score_tolerance=copy.deepcopy(queue.SCORE_TOLERANCE),
        original_metric_reexecution_tolerance=copy.deepcopy(queue.METRIC_TOLERANCE),
        original_range_parameters=dict(slidingWindow=100,thre=250,version='opt'),replay_outputs={},
        fixed_release_checkpoint_evaluation=True,completed_training_seed_slots=0)
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_native_model_replay_passed=True)
    if failure=='resource_abort':receipt['resource_guard_aborted']=True
    elif failure=='nonzero_exit':receipt['exit_code']=1
    elif failure=='short_points':report['full_score_points']=dict(train=2,validation=2,test=1)
    elif failure=='dropped_weights':report['original_full_state_dictionary_loaded_strictly']=False
    elif failure=='missing_spot':report['original_SPOT_init_model_inference_and_threshold_reexecution_passed']=False
    elif failure=='changed_range_grid':report['original_range_parameters']['thre']=10
    elif failure=='looser_scores':report['original_inference_score_tolerance']['rtol']=.1
    elif failure=='looser_metrics':report['original_metric_reexecution_tolerance']['rtol']=.1
    elif failure=='added_seed':report['additional_training_seed_slots']=1
    elif failure=='release_as_training':report['completed_training_seed_slots']=1
    for folder,name,values in [('native','full_native_model_replay.json',report),('state','receipt.json',receipt)]:
        directory=tmp_path/folder;directory.mkdir();(directory/name).write_text(json.dumps(values),encoding='utf-8')
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    if failure:
        with pytest.raises(ValueError):queue.native_terminal(config,job)
    else:assert queue.native_terminal(config,job)==(report,receipt)
