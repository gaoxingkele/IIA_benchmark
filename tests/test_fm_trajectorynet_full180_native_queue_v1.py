import copy
import json
import math
from pathlib import Path
import pytest
from scripts.flow_matching import run_trajectorynet_EB_native_full180_v1 as s


def test_all_original_grid_conditions_full_budget_and_unchanged_guards_are_registered():
    c=s.registration();author=s.read(s.ROOT/s.AUTHOR);plan=s.read(s.ROOT/s.PLAN)
    assert c['available_training_models']==len(c['jobs'])==135
    assert c['deferred_energy_models']==len(c['deferred_energy_jobs'])==45
    assert c['required_grid_models']==180
    assert len({j['id'] for j in c['jobs']})==135
    assert {(j['recipe']['density_weight'],j['recipe']['velocity_weight']) for j in c['jobs']}=={
        (d,v) for d in [0.,.1,.01] for v in [0.,.001,.0001]}
    for index in range(9):
        assert {(j['seed'],j['leftout_timepoint']) for j in c['jobs'] if j['recipe_index']==index}=={
            (seed,t) for seed in [42,43,44,45,46] for t in [1,2,3]}
    assert c['jobs'][0]['recipe']['source_flags']==dict(top_k_reg=.1,vecint=0.,use_magnitude=False)
    assert c['actual_optimizer_steps_per_training_model']==10000 and c['batch_size']==1000
    assert c['source_parameters']['niters']==10000 and c['source_parameters']['batch_size']==1000
    assert c['native_probe_recipe_leaveout_cases']==27
    assert c['native_probe_full_population_leftouts']==[1,2,3]
    assert c['settings']==author['settings'] and c['resource_lock']==author['resource_lock']
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
    assert not c['fixed_test_coefficient_selection']
    assert all(j['recipe_index']>=9 for j in c['deferred_energy_jobs'])


def test_actual_checkpoint_success_report_guard_and_configuration_are_required(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    author=tmp_path/s.AUTHOR;author.parent.mkdir(parents=True);author.write_text('{}')
    report=tmp_path/'native.json';guard=tmp_path/'guard.json'
    c=dict(author_replay_guard='guard.json',author_replay_report=str(report),full_population_counts={'0':2381})
    assert not s.author_ready(c)
    guard.write_text(json.dumps(dict(passed=True,exit_code=0,resource_guard_aborted=True)))
    with pytest.raises(ValueError,match='Preserved'):s.author_ready(c)
    data=dict(passed=True,configuration_sha256=s.sha(author),parameters=9621,
        full_population_counts=c['full_population_counts'],all_source_target_float32_coordinates_equal=True)
    report.write_text(json.dumps(data))
    guard.write_text(json.dumps(dict(passed=True,exit_code=0,resource_guard_aborted=False,report_sha256=s.sha(report))))
    assert s.author_ready(c)
    data['parameters']=9093;report.write_text(json.dumps(data))
    guard.write_text(json.dumps(dict(passed=True,exit_code=0,resource_guard_aborted=False,report_sha256=s.sha(report))))
    with pytest.raises(ValueError,match='Actual full'):s.author_ready(c)


def rows():
    plan=s.read(s.ROOT/s.PLAN)
    jobs=[j for j in s.jobs_for(plan) if j['recipe_index']==3]
    records=[]
    for j in jobs:
        v=(j['seed']-42)+j['leftout_timepoint']/10
        selections={name:dict(primary_W1=v+(1 if name=='source_last_scheduled_validation' else 0))
            for name in ['corrected_validation_minimum','source_last_scheduled_validation']}
        records.append(dict(job=j,passed=True,actual_training_updates=10000,checkpoint_selections=selections))
    return jobs,records


def test_complete_row_uses_five_seed_means_and_keeps_both_checkpoint_selections():
    jobs,records=rows();value=s.aggregate(records,jobs)
    assert value['full_models']==15 and value['seed_count']==5
    a=value['checkpoint_selections']['corrected_validation_minimum']
    b=value['checkpoint_selections']['source_last_scheduled_validation']
    assert math.isclose(a['mean_W1'],2.2) and math.isclose(b['mean_W1'],3.2)
    assert math.isclose(a['five_seed_mean_SD'],math.sqrt(2.5))
    assert not math.isclose(a['five_seed_mean_SD'],a['separate_all15_model_SD'])
    assert not value['selected_hyperparameters_by_test']
    assert not value['paper_selected_coefficient_identity_certified']


def test_partial_leaveouts_duplicate_conditions_mixed_recipes_or_short_training_cannot_be_aggregated():
    jobs,records=rows()
    with pytest.raises(ValueError):s.aggregate(records[:-1],jobs)
    wrong=copy.deepcopy(records);wrong[-1]=wrong[0]
    with pytest.raises(ValueError):s.aggregate(wrong,jobs)
    wrong=copy.deepcopy(records);wrong[0]['actual_training_updates']=9999
    with pytest.raises(ValueError,match='10000'):s.aggregate(wrong,jobs)
    wrong_jobs=copy.deepcopy(jobs);wrong_jobs[0]['recipe_index']=0
    with pytest.raises(ValueError,match='coefficient'):s.aggregate(records,wrong_jobs)
