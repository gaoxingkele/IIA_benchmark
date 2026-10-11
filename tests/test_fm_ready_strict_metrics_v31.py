import copy
import hashlib
import io
import zipfile

import numpy as np
import pytest

from scripts.flow_matching.publish_ready_strict_metrics_v31 import (
    POINT_METRICS, PROTOCOL, array_shape, audit_points, independent_points,
    point_updates, verify_delta,
)


class Arrays(dict):
    @property
    def files(self):
        return list(self)


def original_rows(n=0, required=3):
    return [dict(task='anomaly_detection',algorithm='baseline',dataset='SMD',protocol=PROTOCOL,
        metric=m,n=str(n),required_repeats=str(required),mean='',std='',se='',ci95_low='',ci95_high='',
        source='original',captured_utc='2026-01-01',author_equivalence_certified='False',strict_TAB_result='False')
        for m in POINT_METRICS]


def record(seed, scores=(.1,.2,.6,.8)):
    return dict(algorithm='baseline',dataset='SMD',seed=seed,
        point_audit=dict(micro=independent_points(np.array([0,0,1,1]),np.array(scores),.5)))


def test_unique_training_seeds_and_original_budget_required():
    rows=original_rows()
    with pytest.raises(ValueError,match='twice'):
        point_updates(rows,[record(1),record(1)],'audit','2026-02-01')
    with pytest.raises(ValueError,match='budget'):
        point_updates(original_rows(required=1),[record(1),record(2)],'audit','2026-02-01')
    values=point_updates(rows,[record(1),record(2)],'audit','2026-02-01')
    assert all(v['n']==2 and v['ci95_low'] is not None for v in values)


def test_fixed_weight_single_evaluation_has_no_seed_uncertainty():
    values=point_updates(original_rows(required=1),[record(42)],'audit','2026-02-01')
    assert all(v['n']==1 and v['std'] is None and v['se'] is None and v['ci95_low'] is None for v in values)


def test_unrelated_author_rows_budget_and_identities_cannot_change():
    original=original_rows()
    author=dict(original[0],protocol='author_threshold_PA',mean='.99',n='1')
    original.append(author)
    updates=point_updates(original_rows(),[record(1)],'audit','2026-02-01')
    merged=updates+[author]
    assert len(verify_delta(original,merged,updates))==5
    bad=copy.deepcopy(merged);bad[-1]['mean']='.9'
    with pytest.raises(ValueError):verify_delta(original,bad,updates)
    bad=copy.deepcopy(merged);bad[0]['required_repeats']='4'
    with pytest.raises(ValueError):verify_delta(original,bad,updates)
    with pytest.raises(ValueError):verify_delta(original,merged[:-1],updates)


def fixture():
    validation=np.array([0.,.1,.2,.3])
    test=np.array([0.,.4,.2,.9])
    labels=np.array([0,1,0,1])
    arrays=Arrays(e_train=np.array([0.,.1]),e_validation=validation,e_test=test,e_labels=labels)
    prepared=Arrays(e_labels=labels)
    dataset=dict(input_sha256='a'*64,entities=[dict(array_prefix='e',entity_id='entity',test_points=4)])
    threshold=float(np.percentile(validation,99,method='linear'))
    metrics=independent_points(labels,test,threshold)
    result=dict(checkpoint_sha256='b'*64,metrics={'strict':{'1':dict(protocol='strict_validation_only_no_pa_v1',
        coverage='complete',point_adjustment=False,calibration=dict(calibration_split='validation',
        false_alarm_percent=1,rule='validation_score_percentile_linear_strict_greater',threshold=threshold,
        calibration_dataset_sha256='a'*64,model_artifact_sha256='b'*64,calibration_points=4,
        calibration_sha256=hashlib.sha256(validation.astype('<f8').tobytes()).hexdigest()),
        per_entity={'entity':metrics},micro=metrics,macro_f1=metrics['f1'])}})
    buffer=io.BytesIO()
    np.savez(buffer,e_train=np.zeros((2,3)),e_validation=np.zeros((4,3)),e_test=np.zeros((4,3)),e_labels=labels)
    buffer.seek(0)
    return arrays,prepared,zipfile.ZipFile(buffer),dataset,result


def test_every_original_timestamp_and_validation_array_is_audited():
    arrays,prepared,archive,dataset,result=fixture()
    with archive:
        proof=audit_points(arrays,prepared,archive,dataset,result)
        assert proof['full_test_points']==4 and proof['validation_points']==4
        assert array_shape(archive,'e_validation')==(4,3)
        shortened=Arrays(arrays);shortened['e_test']=shortened['e_test'][:-1]
        with pytest.raises(ValueError,match='timestamp'):
            audit_points(shortened,prepared,archive,dataset,result)
        swapped=Arrays(arrays);swapped['e_validation']=swapped['e_test']
        with pytest.raises(ValueError,match='validation scores'):
            audit_points(swapped,prepared,archive,dataset,result)


@pytest.mark.parametrize('change',['PA','test_calibration','labels','metric','missing_array','threshold'])
def test_protocol_and_complete_arrays_cannot_be_relaxed(change):
    arrays,prepared,archive,dataset,result=fixture()
    strict=result['metrics']['strict']['1']
    if change=='PA':strict['point_adjustment']=True
    elif change=='test_calibration':strict['calibration']['calibration_split']='test'
    elif change=='labels':prepared['e_labels']=np.zeros(4)
    elif change=='metric':strict['micro']=dict(strict['micro'],f1=.99)
    elif change=='missing_array':del arrays['e_train']
    elif change=='threshold':strict['calibration']['threshold']=.99
    with archive,pytest.raises(ValueError):audit_points(arrays,prepared,archive,dataset,result)


def test_undefined_ranking_metric_stays_unavailable():
    metrics=independent_points(np.zeros(3),np.array([.1,.2,.3]),.2)
    assert metrics['auroc'] is None and metrics['average_precision'] is None
    assert metrics['f1']==0
