import pytest
from scripts.flow_matching import audit_otcfm_EB_CITE_full150_replay_v1 as s
from scripts.flow_matching.otcfm_EB_CITE_result_integrity_v1 import full_metric_names


def test_every_original_EB_interior_distribution_is_replayed_without_truncating_to_two():
    actual=s.interior_geometry([2656,4374,3650,3873,3650])
    assert actual==[(2656,4374),(4374,3650),(3650,3873)]
    assert len(actual)*2==6 and len(full_metric_names(5))==44


def test_every_original_CITE_population_is_kept_without_equalizing_or_subsampling():
    assert s.interior_geometry([7476,6999,9511,7254])==[(7476,6999),(6999,9511)]
    assert len(full_metric_names(4))==36


@pytest.mark.parametrize('sizes',[[10,20,30],[10,0,30,40],[10,20,-1,40,50],[10,20,30,40,50,60]])
def test_missing_or_invalid_timepoint_populations_are_rejected(sizes):
    with pytest.raises(ValueError):s.interior_geometry(sizes)


def test_all150_cases_and_original120_replay_guards_remain_intact():
    config=s.registration();q=s.read(s.ROOT/config['original_queue'])
    assert len(config['required_case_ids'])==len(set(config['required_case_ids']))==150
    assert config['required_case_ids']==[j['id'] for j in q['jobs']]
    assert config['settings']==q['settings'] and config['native_metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert config['independent_source_replay_queue_preserved']=='configs/reproducibility/fm_otcfm_singlecell_full120_checkpoint_replay.v2.json'
    assert config['EB_complete_metric_count']==44 and config['CITE_complete_metric_count']==36
    assert config['resource_lock']=='experiments/runs/fm_otcfm_singlecell_readonly_replay_cpu.lock'
