import copy
import itertools
import math
import pytest
from scripts.flow_matching import audit_common105_test_label_oracle_v1 as s

def f1(y,p):
    tp=sum(a and b for a,b in zip(y,p));fp=sum(not a and b for a,b in zip(y,p));fn=sum(a and not b for a,b in zip(y,p))
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.

def test_boundary_touching_anomalies_never_merge_and_oracle_cannot_use_per_entity_thresholds():
    entities={'a':([0,1],[.2,.1]),'b':([1,0],[.9,.3])}
    value=s.grouped_oracle(entities)
    assert value['PA']['f1']<1
    assert s.exact.oracle([.2,.1,.9,.3],[0,1,1,0])['PA']['f1']==1
    assert value['all_test_points']==4 and value['test_anomaly_points']==2
    assert not value['per_entity_threshold_optimization']

def test_grouped_tied_scores_match_exhaustive_original_entity_PA_for_every_binary_labeling():
    x=[.1,.3,.3,.2,.4,.1]
    for y in itertools.product([0,1],repeat=6):
        entities={'a':(y[:3],x[:3]),'b':(y[3:],x[3:])};value=s.grouped_oracle(entities)
        thresholds=[max(x)]+[math.nextafter(v,-math.inf) for v in set(x)]
        for name,pa in [('raw',False),('PA',True)]:
            candidates=[]
            for t in thresholds:
                p=[]
                for labels,scores in entities.values():
                    q=[v>t for v in scores]
                    p+=s.exact.source.point_adjust(labels,q) if pa else q
                candidates.append(f1(y,p))
            assert value[name]['f1']==max(candidates)
            assert value[name]['test_labels_used_to_select_threshold']
            assert value[name]['diagnostic_only'] and not value[name]['deployable_validation_only_result']

def test_complete_registered_scope_and_distinct_light_score_profile_do_not_modify_parent_gates():
    c=s.registration();old=s.read(s.ROOT/s.upstream.CONFIG)
    assert len(c['cases'])==577 and c['required_label_view_results']==1726
    assert c['cases']==old['cases'] and c['resource_lock']==old['resource_lock']
    assert c['settings']['minimum_commit_headroom_bytes']==16*2**30
    assert old['settings']['minimum_commit_headroom_bytes']==28*2**30
    assert c['settings']['maximum_worker_private_bytes']==int(1.5*2**30)
    assert c['inter_case_yield_seconds']==20
    assert c['test_labels_used_to_select_oracle_threshold'] and c['oracle_scores_never_enter_strict_leaderboards']
    assert not c['fixed_original_validation_controls_test_label_threshold_selection']
    changed=copy.deepcopy(c);changed['test_labels_used_to_select_oracle_threshold']=False
    with pytest.raises(ValueError,match='All577'):s.verify(changed)

@pytest.mark.parametrize('entities',[{}, {'a':([0,.5],[.1,.2])}, {'a':([0,1],[.1])}, {'a':([0,1],[.1,float('nan')])}])
def test_incomplete_or_nonbinary_or_nonfinite_original_populations_rejected(entities):
    with pytest.raises(ValueError):s.grouped_oracle(entities)
