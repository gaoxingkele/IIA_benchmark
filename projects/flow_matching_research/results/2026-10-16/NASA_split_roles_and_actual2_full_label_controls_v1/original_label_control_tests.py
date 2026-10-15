from copy import deepcopy
import numpy as np
import pytest
from scripts.flow_matching import audit_timesnet_declared_CSV_label_control_v1 as s


def test_all51entities_and_five_cases_keep_original_model_and_evaluation_settings():
    c=s.registration();s.verify(c);old=s.previous.registration()
    for key in ['jobs','settings','resource_lock','source_evaluation','label_policies','environment','python']:
        assert c[key]==old[key]
    assert c['representation_evidence']['input_values_only_used_for_registration']
    assert not c['old_direct_float64_gate_changed'] and not c['model_threshold_and_metric_math_changed']
    assert c['declared_data_representation_changed']


@pytest.mark.parametrize('field',['jobs','declared_representation_control','label_policies','settings','old_direct_float64_gate_changed'])
def test_any_dropped_case_other_transform_or_relaxed_old_gate_rejected(field):
    c=s.registration()
    if field in ['jobs','label_policies']:c[field].pop()
    elif field=='settings':c[field]['minimum_commit_headroom_bytes']=1
    elif field=='declared_representation_control':c[field]['channel']='all_channels'
    else:c[field]=True
    with pytest.raises(ValueError):s.verify(c)


def test_declared_G7_control_keeps_old_direct_gate_and_never_mutates_raw():
    raw=np.array([[.12345678901234567],[.3816289182780123],[1.],[0.],[.1],[.2],[.3],[.4]])
    before=raw.copy();represented=s.characterization.representation(raw,s.PROFILE)
    rows=[dict(chan_id='G-7',num_values='8',anomaly_sequences='[[1, 3]]')]
    timestamps=np.arange(1,9)
    old,_=s.previous.original.mapped_labels(represented,timestamps,raw,rows)
    report,labels=s.mapped_labels(represented,timestamps,raw,rows)
    assert report['old_direct_mapping_proven']==old['full_mapping_proven']
    assert report['full_mapping_proven'] and labels is not None
    assert report['declared_representation_profile']==s.PROFILE and np.array_equal(raw,before)
    assert not report['tolerance_used'] and not report['representation_selected_by_model_performance']


def test_other50entities_retain_direct_values_even_if_transform_would_match():
    raw=np.array([[.12345678901234567],[.3816289182780123],[1.],[0.],[.1],[.2],[.3],[.4]])
    rows=[dict(chan_id='P-1',num_values='8',anomaly_sequences='[[1, 3]]')]
    report,labels=s.mapped_labels(raw,np.arange(1,9),raw,rows)
    assert report['full_mapping_proven'] and report['declared_representation_profile']=='direct_float64'
    assert np.array_equal(raw,raw.copy())


def test_same_saved_score_worker_and_protocol_math_are_reused():
    private=s.private_module()
    assert private.worker.__code__.co_code==s.previous.original.worker.__code__.co_code
    assert private.protocol_diagnostics.__code__.co_code==s.previous.original.protocol_diagnostics.__code__.co_code
    assert private.mapped_labels is s.mapped_labels
