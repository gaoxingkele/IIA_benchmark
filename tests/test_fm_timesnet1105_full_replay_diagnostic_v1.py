import ast
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.flow_matching import diagnose_timesnet1105_full_replay_v1 as s


def test_exact_equality_and_one_ULP_difference_are_not_conflated():
    values=np.array([1.,2.,3.])
    assert s.compare_scores(values,values)['scores_exactly_equal']
    changed=values.copy();changed[1]=np.nextafter(changed[1],np.inf)
    result=s.compare_scores(changed,values)
    assert not result['scores_exactly_equal'] and result['changed_points']==1
    assert result['maximum_absolute_difference']>0


@pytest.mark.parametrize('values',[np.array([1.,np.nan]),np.array([1.,np.inf]),np.array([1.])])
def test_incomplete_or_nonfinite_coordinates_still_fail(values):
    with pytest.raises(ValueError):s.compare_scores(values,np.array([1.,2.]))


def test_fixed_threshold_controls_keep_strict_greater_and_no_PA():
    y=np.array([0,1,1,0]);score=np.array([.5,.5,.6,.8])
    r=s.fixed_threshold_counts(y,score,.5)
    assert r['tp']==1 and r['fp']==1 and r['fn']==1 and r['F1']==.5


def test_original_failure_is_admitted_only_as_preserved_terminal_mismatch(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    attempt=tmp_path/'attempt';attempt.mkdir()
    receipt=dict(exit_code=1,passed=False,resource_guard_aborted=False,proof_sha256=None)
    (attempt/'receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
    (attempt/'stderr.log').write_text(s.ERROR,encoding='utf-8')
    stage=dict(state_root='attempt',output_directory=str(tmp_path/'out'))
    proof=s.preserved_failure(stage)
    assert proof['strict_failure_remains']
    receipt['resource_guard_aborted']=True
    (attempt/'receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
    with pytest.raises(ValueError):s.preserved_failure(stage)


def test_all51_full_original_geometry_caps_and_point_success_are_bound():
    c=s.registration();stage=s.verify(c)
    assert c['required_entities']==51 and c['required_test_points']==408160
    assert c['required_train_points']==101565 and c['required_validation_points']==26684
    assert c['full_original_batch_size']==128 and c['full_original_parameters']==28132633
    assert c['settings']==stage['settings'] and c['resource_lock']==stage['resource_lock']
    assert c['cpu_threads']==stage['cpu_threads']==2
    assert c['settings']['minimum_commit_headroom_bytes']==28*2**30
    assert c['new_training_updates']==0 and c['preserved_failure']['strict_failure_remains']
    changed=copy.deepcopy(c);changed['required_entities']=1
    with pytest.raises(ValueError):s.verify(changed)


def test_worker_has_no_fit_updates_or_subsample_and_cannot_certify_old_failure():
    tree=ast.parse(Path(s.__file__).read_text('utf-8'))
    worker=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
    assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
        and x.func.attr in ['fit','backward','step'] for x in ast.walk(worker))
    assert any(isinstance(x,ast.For) and ast.unparse(x.iter)=="dataset['entities']" for x in ast.walk(worker))
    claims=[x for x in ast.walk(worker) if isinstance(x,ast.keyword) and x.arg=='strict_native_reproduction_certified']
    assert len(claims)==2 and all(ast.literal_eval(x.value) is False for x in claims)
    assert any(isinstance(x,ast.keyword) and x.arg=='strict_historical_failure_preserved'
        and ast.literal_eval(x.value) is True for x in ast.walk(worker))
