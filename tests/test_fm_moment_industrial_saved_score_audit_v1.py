import ast
from copy import deepcopy
import inspect

import numpy as np
import pytest

from scripts.flow_matching import audit_moment_industrial_saved_scores_v1 as audit


class Archive(dict):
    @property
    def files(self):
        return list(self)


def inputs():
    dataset = dict(training_groups=[dict(group_id='normal', array_prefix='t', features=2, points=5)],
        validation_groups=[dict(group_id='calibration', array_prefix='v', features=2, points=4)],
        entities=[dict(group_id='experiment', entity_id='experiment[0:6]', array_prefix='e',
                       features=2, points=6, test_points=6)])
    raw = Archive(t_values=np.ones((5,2)), v_values=np.ones((4,2)), e_test=np.ones((6,2)),
                  e_labels=np.array([0,0,1,1,0,0]), e_source_rows=np.arange(6))
    scores = Archive(t_train=np.arange(5,dtype=float), v_validation=np.arange(4,dtype=float),
                     e_test=np.arange(6,dtype=float), e_labels=raw['e_labels'].copy(),
                     e_source_rows=raw['e_source_rows'].copy())
    coverage = [dict(group=entry.get('entity_id', entry['group_id']),role=role,source_array=key,
                     native_points=len(raw[key]),all_context_tokens_observed=True)
                for role,entry,key in audit.groups(dataset)]
    return dataset,raw,scores,dict(native_coverage=coverage)


def test_complete_separate_groups_and_metadata_accepted():
    dataset,raw,scores,result = inputs()
    entities,lengths,validation = audit.check_industrial_arrays(raw,scores,dataset)
    assert lengths == {'experiment[0:6]':6}
    assert np.array_equal(validation,scores['v_validation'])
    assert np.array_equal(entities['experiment[0:6]'][0],raw['e_labels'])
    audit.check_industrial_coverage(raw,result,dataset)


@pytest.mark.parametrize('fault',['short_train','short_test','nonfinite','labels','metadata',
                                  'extra_array','missing_array','features','points','shared_group','prefix'])
def test_partial_or_changed_group_arrays_are_rejected(fault):
    dataset,raw,scores,_ = inputs()
    if fault == 'short_train': scores['t_train']=scores['t_train'][:-1]
    elif fault == 'short_test': scores['e_test']=scores['e_test'][:-1]
    elif fault == 'nonfinite': scores['e_test'][1]=np.nan
    elif fault == 'labels': scores['e_labels'][2]=0
    elif fault == 'metadata': scores['e_source_rows'][2]=3
    elif fault == 'extra_array': scores['unexpected']=np.zeros(3)
    elif fault == 'missing_array': del scores['e_source_rows']
    elif fault == 'features': dataset['entities'][0]['features']=1
    elif fault == 'points': dataset['entities'][0]['points']=5
    elif fault == 'shared_group': dataset['validation_groups'][0]['group_id']='normal'
    elif fault == 'prefix': dataset['validation_groups'][0]['array_prefix']='t'
    with pytest.raises((ValueError,KeyError)):
        audit.check_industrial_arrays(raw,scores,dataset)


@pytest.mark.parametrize('fault',['missing','duplicate','source','points','mask','role'])
def test_wrong_or_incomplete_native_coverage_is_rejected(fault):
    dataset,raw,_,result = inputs()
    rows=result['native_coverage']
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows.append(deepcopy(rows[0]))
    elif fault == 'source': rows[0]['source_array']='other_values'
    elif fault == 'points': rows[0]['native_points']-=1
    elif fault == 'mask': rows[0]['all_context_tokens_observed']=False
    elif fault == 'role': rows[0]['role']='validation'
    with pytest.raises(ValueError):
        audit.check_industrial_coverage(raw,result,dataset)


def test_metric_bootstrap_range_and_terminal_checks_remain_original():
    before=ast.parse(inspect.getsource(audit.original.worker))
    after=audit.worker_ast()
    functions={'evaluate_saved','pointwise_metrics','point_adjust','entity_buffers','verify_macros','validate_identity'}
    def scientific_calls(tree):
        return sorted(ast.dump(node) for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id in functions)
    assert scientific_calls(before)==scientific_calls(after)
    unchanged_messages=['Use the actual producer Python/NumPy/Torch environment without widening tolerances',
        'Every primary/core diagnostic metric and grouped uncertainty must recompute exactly',
        'Full original range evaluation must bind exact arrays and scientific configuration',
        'Actual completed guarded full range evaluation required']
    strings={node.value for node in ast.walk(after) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
    assert all(message in strings for message in unchanged_messages)
