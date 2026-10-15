from copy import deepcopy
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v1 as s


def metadata():
    return [dict(num_values='8',anomaly_sequences='[[1, 3]]'),dict(num_values='8',anomaly_sequences='[[4, 6]]')]


def test_five_original_seeds_all_entities_and_all_four_label_policies_required():
    c=s.registration();jobs=s.verify(c)
    assert len(jobs)==5 and {j['seed'] for j in jobs}==set(range(1103,1108))
    assert c['entities_per_model']==51 and c['test_points_per_model']==408160 and len(c['label_policies'])==4
    original=s.read(s.ROOT/s.audit.GUARD)
    assert c['settings']==original['settings'] and c['resource_lock']==original['resource_lock']
    assert c['exact_coordinate_mapping_required'] and c['no_tolerance_or_unregistered_offset_permitted']


@pytest.mark.parametrize('field',['jobs','label_policies','settings','exact_coordinate_mapping_required'])
def test_partial_cohort_selected_labels_relaxed_resources_or_approximate_mapping_rejected(field):
    c=s.registration()
    if field in ['jobs','label_policies']:c[field].pop()
    elif field=='settings':c[field]['minimum_commit_headroom_bytes']=1
    else:c[field]=False
    with pytest.raises(ValueError):s.verify(c)


def test_every_coordinate_must_match_before_transferring_any_labels():
    raw=np.arange(16,dtype=float).reshape(8,2);values=raw[2:5].copy();timestamps=np.arange(3,6)
    report,views=s.mapped_labels(values,timestamps,raw,metadata())
    assert report['full_mapping_proven'] and report['exact_source_mappings']==1 and not report['tolerance_used']
    assert views['source_first_exclusive'].tolist()==[1,0,0]
    assert views['source_last_inclusive'].tolist()==[0,0,1]
    values[1,0]+=1e-12
    report,views=s.mapped_labels(values,timestamps,raw,metadata())
    assert not report['full_mapping_proven'] and views is None


def test_ambiguous_identical_features_with_different_labels_cannot_choose_time_origin():
    report,views=s.mapped_labels(np.ones((3,1)),np.arange(2,5),np.ones((8,1)),metadata())
    assert report['exact_source_mappings']==2 and report['timestamp_ambiguity_changes_labels']
    assert not report['full_mapping_proven'] and views is None


def test_two_coordinate_mappings_with_identical_all_label_views_keep_both_candidates():
    rows=[dict(num_values='8',anomaly_sequences='[[6, 7]]')]
    report,views=s.mapped_labels(np.ones((3,1)),np.arange(2,5),np.ones((8,1)),rows)
    assert report['exact_source_mappings']==2 and report['full_mapping_proven']
    assert not report['offset_inferred_or_test_selected'] and all(not a.any() for a in views.values())


@pytest.mark.parametrize('fault',['duplicate_time','fractional_time','feature_width','reverse_time'])
def test_invalid_timestamp_or_feature_identity_rejected(fault):
    raw=np.ones((8,1));values=np.ones((3,1));times=np.array([1.,2.,3.])
    if fault=='duplicate_time':times[1]=1
    elif fault=='fractional_time':times[1]=1.5
    elif fault=='feature_width':values=np.ones((3,2))
    else:times=times[::-1]
    with pytest.raises(ValueError):s.mapped_labels(values,times,raw,metadata())


def test_PA_respects_entity_boundaries_and_does_not_mutate_fixed_scores():
    entities={'a':(np.array([0,1,1]),np.array([0.,0.,0.])),
              'b':(np.array([1,1,0]),np.array([1.,0.,0.]))}
    before=deepcopy(entities);result=s.protocol_diagnostics(entities,.5)
    assert result['validation_only_noPA']['true_positives']==1
    assert result['same_validation_threshold_entitywise_PA_diagnostic']['true_positives']==2
    assert result['always_anomaly_noPA_F1']==pytest.approx(2*4/(6+4))
    assert all(np.array_equal(value,before[k][i]) for k,items in entities.items() for i,value in enumerate(items))
    assert not result['PA_oracle_fields_are_strict_leaderboard_metrics']


@pytest.mark.parametrize('fault',['nonfinite_score','invalid_label','incomplete_score','nonfinite_threshold'])
def test_incomplete_or_invalid_diagnostic_cannot_publish_metrics(fault):
    y=np.array([0,1]);scores=np.array([0.,1.]);threshold=.5
    if fault=='nonfinite_score':scores[0]=np.nan
    elif fault=='invalid_label':y[0]=2
    elif fault=='incomplete_score':scores=scores[:1]
    else:threshold=np.nan
    with pytest.raises(ValueError):s.protocol_diagnostics({'a':(y,scores)},threshold)


def test_source_native_model_proof_required_before_any_numerical_audit(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'source_native',lambda *a:None)
    with pytest.raises(ValueError,match='before numerical imports'):s.worker({},dict(seed=1103),tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


def test_cold_registration_does_not_import_numpy_or_torch():
    code='from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
