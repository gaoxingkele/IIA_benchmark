from copy import deepcopy
import numpy as np
import pytest
from scripts.flow_matching import audit_full873_window_geometry_routing_v2 as s


def row(family,window):
    entity=dict(entity_id='fixture',array_prefix='e',features=2,fit_interval=[0,112],validation_interval=[212,312],test_points=200)
    headers={'e_'+r:dict(shape=[n,2]) for r,n in [('train',112),('validation',100),('test',200)]}
    return s.prior.inspect_case(dict(family=family,window=window),dict(entities=[entity]),headers)


def test_specialized_worker_does_not_fail_the_generic512_length_gate():
    original=row('moment',512);actual=s.classify(original,dict(verified_route=True))
    assert original['training_rejected_before_fit'] and original['incompatible_role_inputs']
    assert actual['correction_to_v1'] and not actual['actual_worker_predicted_incompatible']
    assert not actual['predicted_rejection_before_fit'] and not actual['predicted_rejection_after_fit']
    assert not actual['actual_job_terminal_failure_observed']


def test_DCdetector_still_needs_a_real105point_validation_source():
    actual=s.classify(row('baselines',105),dict(verified_route=True))
    assert actual['actual_worker_predicted_incompatible'] and actual['predicted_rejection_after_fit']
    assert not actual['predicted_rejection_before_fit'] and not actual['correction_to_v1']


@pytest.mark.parametrize('length',[100,105,107,239,511])
def test_real_specialized_source_function_accepts_short_data_with_observed_edge_fill(length):
    import ast
    path=s.ROOT/'scripts/flow_matching/run_moment_pretrained_job.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='segment_windows')
    namespace=dict(np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    data=np.arange(length*3).reshape(length,3)
    windows,starts,padding=namespace['segment_windows'](data,512)
    assert windows.shape==(1,512,3) and np.array_equal(windows[0,:length],data)
    assert np.array_equal(windows[0,length:],np.repeat(data[-1:],512-length,axis=0))
    assert padding==512-length and np.array_equal(starts,[0])


def test_actual_registered_controller_routes_to_verified_specialized_worker():
    scalar=s.read(s.ROOT/s.scalar.CONFIG);execution=s.read(s.ROOT/scalar['source_execution'])
    route=s.actual_moment_route(execution)
    assert route['fill_tokens_observed'] and not route['fill_targets_scored']
    assert not route['generic_short_sequence_rejection_applies']
    bad=deepcopy(execution)
    next(f for f in bad['families'] if f['name']=='moment')['controller_source']='scripts/flow_matching/run_tsad_queue.py'
    with pytest.raises(ValueError):s.actual_moment_route(bad)
