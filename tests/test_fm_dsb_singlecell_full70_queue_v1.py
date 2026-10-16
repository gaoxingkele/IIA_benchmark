import math
import pytest
from scripts.flow_matching import run_dsb_singlecell_full70_v1 as s


def test_every_available_and_known_deferred_case_seed_is_retained():
    c=s.registration()
    assert (c['required_models'],c['available_models'],c['deferred_models'])==(70,60,10)
    assert len(c['jobs'])==60 and len(c['deferred_jobs'])==10
    alljobs=c['jobs']+c['deferred_jobs']
    assert len({j['id'] for j in alljobs})==70
    assert {j['seed'] for j in alljobs}=={42,43,44,45,46}
    assert len({j['dataset'] for j in c['jobs']})==5
    assert {j['dataset'] for j in c['deferred_jobs']}=={s.DEFERRED}
    assert all(j['data'] is None for j in c['deferred_jobs'])


def test_original_preflight_guard_environment_and_full_budgets_are_required():
    c=s.registration();parent=s.read(s.ROOT/s.CORE)
    for key in ['settings','resource_lock','python','environment_lock']:
        assert c[key]==parent[key]
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
    assert (c['actual_updates_per_model'],c['training_batch'],c['cache_particles'],c['langevin_steps'])==(400040,512,10000,20)
    assert c['required_native_constructor_probe_cases_available']==12
    assert c['required_native_constructor_probe_cases_deferred']==2
    assert c['require_bitwise_stochastic_CPU_checkpoint_replay']
    assert not c['all_original_experiments_complete']


def test_no_partial_seed_or_leftout_cohort_can_be_promoted_to_a_paper_row():
    jobs=s.jobs_for([dict(dataset='fixture',leftout_timepoint=i) for i in [1,2]],[42,43,44,45,46])
    rows=[dict(job=j,native_metrics={'1-Wasserstein':float(j['seed']-42)+2*j['leftout_timepoint']}) for j in jobs]
    row=s.aggregate_complete_row(rows,jobs)
    values=row['metrics']['1-Wasserstein']
    assert row['full_models']==10 and values['mean']==5
    assert math.isclose(values['five_seed_mean_SD'],math.sqrt(2.5))
    assert values['descriptive_df4_t95'][0]<5<values['descriptive_df4_t95'][1]
    assert not row['historical_paper_equivalence_certified']
    with pytest.raises(ValueError,match='partial'):
        s.aggregate_complete_row(rows[:-1],jobs)


def test_raw_and_core_inputs_are_bound_without_native_performance_claims():
    c=s.registration();paths={r['path'] for r in c['source_receipts']}
    assert s.CORE in paths and s.ADAPTER in paths and s.COMPONENTS in paths
    assert s.ACQUISITION in paths and c['overlay_receipt'] in paths
    assert c['paper_targets']['EB']['mean']==.862
    assert c['paper_targets']['Multiome']['mean']==1.079
    assert c['paper_targets']['CITE']['mean']==.953
