import copy

import pytest

from scripts.flow_matching import run_saits_completion_replay_queue_v1 as queue


def fixtures(monkeypatch):
    jobs=[];native={}
    for i in range(210):
        path='configs/experiments/job%d.json'%i
        native[path]=dict(id='original%d'%i,device='cuda',bootstrap_group='month',
            seed=26+i%5,diagnostic=False,limit_test_cases=None)
        jobs.append(dict(config=path,config_sha256='frozen',runner='scripts/flow_matching/run_saits.py'))
    monkeypatch.setattr(queue,'sha',lambda p:'frozen')
    monkeypatch.setattr(queue,'read',lambda p:native[p.relative_to(queue.ROOT).as_posix()])
    return dict(jobs=jobs),native


def test_all210_original_jobs_and_canonical_seeds_remain(monkeypatch):
    raw,native=fixtures(monkeypatch)
    actual=queue.select_jobs(raw)
    assert len(actual)==210 and set(actual)=={n['id'] for n in native.values()}
    assert {x['native']['seed'] for x in actual.values()}=={26,27,28,29,30}


@pytest.mark.parametrize('change', ['subset','duplicate','diagnostic','limited_test','wrong_device',
    'wrong_seed','wrong_groups','unsafe_id','changed_source','wrong_runner'])
def test_original_scope_or_source_cannot_be_silently_changed(monkeypatch,change):
    raw,native=fixtures(monkeypatch);first=native['configs/experiments/job0.json']
    if change=='subset':raw['jobs'].pop()
    if change=='duplicate':native['configs/experiments/job1.json']['id']=first['id']
    if change=='diagnostic':first['diagnostic']=True
    if change=='limited_test':first['limit_test_cases']=1
    if change=='wrong_device':first['device']='cpu'
    if change=='wrong_seed':first['seed']=0
    if change=='wrong_groups':first['bootstrap_group']='synthetic'
    if change=='unsafe_id':first['id']='../outside'
    if change=='changed_source':raw['jobs'][0]['config_sha256']='changed'
    if change=='wrong_runner':raw['jobs'][0]['runner']='reduced_runner.py'
    with pytest.raises(ValueError):queue.select_jobs(raw)


def completed():
    native=dict(id='canonical',dataset_sha256='frozen',device='cuda')
    result=dict(status='completed',config=copy.deepcopy(native),config_sha256='frozen_config',
        dataset_sha256='frozen',bootstrap_group='month',parameters=11511000,
        test_windows=290,target_count=1944331)
    return native,result


def test_exact_completed_native_result_is_ready():
    queue.check_completed_result(*completed())


@pytest.mark.parametrize('key,value', [('status','running'),('dataset_sha256','changed'),
    ('bootstrap_group','synthetic'),('parameters',0),('test_windows',0),('target_count',0),
    ('benchmark_smoke',True),('diagnostic',True),('config_sha256',None)])
def test_pending_reduced_or_misidentified_result_is_rejected(key,value):
    native,result=completed();result[key]=value
    with pytest.raises(ValueError):queue.check_completed_result(native,result)


def test_completed_result_config_cannot_be_replaced():
    native,result=completed();result['config']['device']='cpu'
    with pytest.raises(ValueError):queue.check_completed_result(native,result)
