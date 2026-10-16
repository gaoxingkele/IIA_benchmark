from copy import deepcopy
import pytest
from scripts.flow_matching import run_otcfm_CITE13176_GEO_native60_v1 as s


def test_all60_original_method_recipe_leaveout_seed_conditions_preserve_previous360():
    cases=s.scientific_cases()
    ids=[j['id'] for j in cases]
    for path in ['configs/experiments/fm_otcfm_singlecell_native_full120.v4.json',
        'configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json',
        'configs/experiments/fm_otcfm_EB100_code_referenced_native_full90.v1.json']:
        ids += [j['id'] for j in s.read(s.ROOT/path)['jobs']]
    assert len(ids)==len(set(ids))==420 and len(cases)==60
    assert {j['seed'] for j in cases}==set(range(42,47)) and {j['leaveout_timepoint'] for j in cases}=={1,2}
    assert {j['method'] for j in cases}=={'cfm','otcfm','sbcfm'}
    assert all(j['scientific_protocol']['max_epochs']==1000 for j in cases)
    assert s.scoped_function('worker').__code__ is s.base.worker.__code__


@pytest.mark.parametrize('fault',['missing_method','missing_leave2','duplicate'])
def test_real_data_preflight_cannot_narrow_original_branches(fault):
    jobs=deepcopy(s.scientific_cases())
    if fault=='missing_method':jobs=[j for j in jobs if j['method']!='sbcfm']
    elif fault=='missing_leave2':jobs=[j for j in jobs if j['leaveout_timepoint']!=2]
    else:jobs.append(deepcopy(jobs[0]))
    with pytest.raises(ValueError):s.preflight_cases(dict(jobs=jobs))


def test_actual_preflight_scope_adaptation_preserves_matching_writer_reader_schema_without_training():
    fn=s.scoped_function('preflight_worker');captured=[]
    fn.__globals__['worker']=lambda q,j,probe:dict(id=j['id'],preflight_only=probe)
    fn.__globals__['sha']=lambda p:'frozen_source60_queue'
    fn.__globals__['exclusive']=lambda p,r:captured.append((p,r))
    jobs=s.scientific_cases()
    for j in jobs:j['validation_compatibility']=dict(only_fixture_not_performance=True)
    fn(dict(jobs=jobs,state_root='fixture_not_a_benchmark'))
    path,report=captured[0]
    assert path.name=='native12_preflight.json' and len(report['records'])==12
    assert report['full60_source_queue_sha256']=='frozen_source60_queue'
    assert report['model_training_updates']==0 and not report['benchmark_performance_claim']
    run=s.scoped_function('run_preflight')
    assert 'full60_source_queue_sha256' in run.__code__.co_consts


def test_deferred_plan_preserves_both_full_budget_tracks_and_original_headroom():
    p=s.plan_registration();old=s.read(s.ROOT/s.base.original.QUEUE)
    assert len(p['planned_cases'])==p['required_models']==60 and p['required_preflight_combinations']==12
    assert p['source_settings']==old['settings'] and p['preserved_existing_singlecell_cases']==360
    assert p['preparation_configuration']==s.preparation.CONFIG
