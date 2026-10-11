import copy
import hashlib
import io
import zipfile

import numpy as np
import pytest

from scripts.flow_matching.publish_ready_strict_metrics_v32 import full_points
from tests.test_fm_ready_strict_metrics_v31 import Arrays,independent_points


def fixture():
    source=dict(t_values=np.zeros((3,2)),v_values=np.zeros((4,2)),e_test=np.zeros((4,2)),
        t_labels=np.zeros(3),v_labels=np.zeros(4),e_labels=np.array([0,1,0,1]),
        t_timestamps=np.arange(3),v_timestamps=np.arange(4)+100,e_timestamps=np.arange(4)+200,
        t_source_rows=np.arange(3),v_source_rows=np.arange(4)+100,e_source_rows=np.arange(4)+200)
    scores=Arrays({k:v.copy() for k,v in source.items() if not k.endswith('_values') and k!='e_test'})
    scores.update(t_train=np.array([0.,.1,.2]),v_validation=np.array([0.,.1,.2,.3]),e_test=np.array([0.,.4,.2,.9]))
    groups=lambda prefix,points,start:dict(array_prefix=prefix,entity_id=prefix,points=points,
        test_points=points,source_path='original.csv',source_interval=[start,start+points])
    dataset=dict(input_sha256='a'*64,grouped_splits_verified=True,training_groups=[groups('t',3,0)],
        validation_groups=[groups('v',4,100)],entities=[groups('e',4,200)])
    threshold=float(np.percentile(scores['v_validation'],99,method='linear'))
    metric=independent_points(source['e_labels'],scores['e_test'],threshold)
    result=dict(checkpoint_sha256='b'*64,metrics={'strict':{'1':dict(protocol='strict_validation_only_no_pa_v1',
        coverage='complete',point_adjustment=False,calibration=dict(calibration_split='validation',false_alarm_percent=1,
        rule='validation_score_percentile_linear_strict_greater',threshold=threshold,calibration_dataset_sha256='a'*64,
        model_artifact_sha256='b'*64,calibration_points=4,
        calibration_sha256=hashlib.sha256(scores['v_validation'].astype('<f8').tobytes()).hexdigest()),
        per_entity={'e':metric},micro=metric,macro_f1=metric['f1'])}})
    memory=io.BytesIO();np.savez(memory,**source);memory.seek(0)
    return scores,Arrays(source),zipfile.ZipFile(memory),dataset,result


def test_original_separate_full_groups_and_all_provenance_are_audited():
    scores,prepared,archive,dataset,result=fixture()
    with archive:
        proof=full_points(scores,prepared,archive,dataset,result)
    assert proof['full_test_points']==4 and proof['validation_points']==4
    assert len(proof['original_grouped_train_validation_test_segments'])==3
    assert proof['every_original_timestamp_source_row_and_fault_label_verified']


@pytest.mark.parametrize('change',['row_order','timestamps','missing_group','short_validation','overlap','PA'])
def test_full_industrial_identity_and_split_checks_cannot_be_relaxed(change):
    scores,prepared,archive,dataset,result=fixture()
    if change=='row_order':scores['e_source_rows']=scores['e_source_rows'][::-1]
    elif change=='timestamps':scores['v_timestamps']=scores['v_timestamps'][::-1]
    elif change=='missing_group':del dataset['validation_groups']
    elif change=='short_validation':scores['v_validation']=scores['v_validation'][:-1]
    elif change=='overlap':dataset['entities'][0]['source_interval']=[100,104]
    elif change=='PA':result['metrics']['strict']['1']['point_adjustment']=True
    with archive,pytest.raises((ValueError,KeyError)):full_points(scores,prepared,archive,dataset,result)
