"""Analytical metrics/formulas, full budgets and real checkpoint replay plumbing."""
from copy import deepcopy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.flow_matching import anomaly_transformer_independent_math_v1 as independent
from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as runner
from scripts.flow_matching.register_anomaly_transformer_full_checkpoint_review_v1 import registration


def test_all500_unique_complete_models_and_four_source_families():
    config = registration(); cases, _ = runner.verify(config)
    assert len(cases) == 500
    assert {k: sum(c['family'] == k for c in cases.values()) for k in runner.FAMILIES} == dict(main=25, sensitivity=400, max=50, plain=25)
    assert len({(c['family'], c['job']['id']) for c in cases.values()}) == 500


@pytest.mark.parametrize('change', ['missing', 'guards', 'tolerance', 'bytes', 'threads', 'scope'])
def test_registration_rejects_incomplete_or_relaxed_review(change):
    c = registration()
    if change == 'missing': c['required_ids'].pop()
    elif change == 'guards': c['settings']['minimum_commit_headroom_bytes'] = 32*1024**3
    elif change == 'tolerance': c['formula_tolerance']['atol'] = 1e-3
    elif change == 'bytes': c['native_array_comparison'] = 'mean_only'
    elif change == 'threads': c['cpu_threads'] = 1
    else: c['original_scope']['papers'] = 1
    with pytest.raises(ValueError): runner.verify(c)


def test_exact_native_bytes_reject_signed_zero_dtype_and_shape_changes():
    independent.require_identical(np.array([0.], np.float32), np.array([0.], np.float32))
    for values in [np.array([-0.], np.float32), np.array([0.], np.float64), np.array([[0.]], np.float32)]:
        with pytest.raises(ValueError): independent.require_identical(np.array([0.], np.float32), values)


def test_head_average_precedes_KL_and_layer_average_precedes_softmax():
    # Both means are uniform, despite large differences between individual heads.
    p = np.array([[[[.9, .1], [.1, .9]], [[.1, .9], [.9, .1]]]], dtype=np.float32)
    s = np.full_like(p, .5)
    x = np.zeros((1, 2, 3), np.float32); target = np.ones_like(x)
    scores, rec, layers = independent.formula_scores(x, target, [p], [s])
    assert np.max(np.abs(layers)) < 1e-12
    np.testing.assert_array_equal(rec, np.ones((1, 2)))
    np.testing.assert_allclose(scores['all.association'], .5)
    np.testing.assert_allclose(scores['all.multiplication'], .5)


def test_plain_entropy_diagnostic_does_not_invent_discrepancy():
    x = np.zeros((1, 2, 3), np.float32); s = np.full((1, 2, 2, 2), .5, np.float32)
    scores, _, entropy = independent.formula_scores(x, x+1, [None], [s], plain=True)
    assert list(scores) == ['all.reconstruction']
    np.testing.assert_allclose(entropy, np.log(2))
    with pytest.raises(ValueError): independent.formula_scores(x, x, [s], [s], plain=True)


def test_tie_aware_AUROC_and_AP_against_hand_calculation():
    auc, ap = independent.ranking_metrics([0, 1, 0, 1], [0., 1., 1., 2.])
    assert auc == .875 and ap == pytest.approx(5/6)
    assert independent.ranking_metrics([1, 1], [1, 2]) == (None, None)
    assert independent.ranking_metrics([0, 1, 0, 1], [1, 1, 1, 1]) == (.5, .5)


def test_validation_only_threshold_strict_greater_and_segment_PA():
    y = np.array([1, 1, 0, 1, 1, 0], np.int8)
    scores = np.array([0., 2., 1., 0., 0., 2.], np.float32)
    result = independent.independent_evaluation(y, scores, np.array([1., 1.], np.float32), 1, .1, [.5, 1, 1.5, 2, 10, 20, 30])
    assert result['validation_ratio']['threshold'] == 1.
    assert result['validation_ratio']['no_PA']['true_positives'] == 1
    assert result['validation_ratio']['paper_PA']['true_positives'] == 2
    changed = independent.independent_evaluation(y, scores*100, np.ones(2, np.float32), 1, .1, [.5, 1, 1.5, 2, 10, 20, 30])
    assert changed['validation_ratio']['threshold'] == 1.
    with pytest.raises(ValueError): independent.require_same_metrics(result, changed)


def test_all_independent_metrics_agree_with_frozen_source_including_ties():
    from scripts.flow_matching.run_anomaly_transformer_paper_contract_v1 import evaluation
    rng = np.random.default_rng(1103)
    y = rng.integers(0, 2, 500, dtype=np.int8)
    scores = rng.integers(0, 9, 500).astype(np.float32)
    validation = rng.integers(0, 9, 330).astype(np.float32)
    args = (y, scores, validation, .5, .1, [.5, 1, 1.5, 2, 10, 20, 30])
    independent.require_same_metrics(evaluation(*args), independent.independent_evaluation(*args))


def epoch_fixture():
    split = dict(train=566724, validation=141681, test=708420)
    rows = [dict(epoch=i, steps=178, train_windows=5667, validation_windows=1416,
                 validation_losses=[1., -1.] if i == 1 else [2., 0.],
                 checkpoint_saved=i == 1, remaining_patience=4-i if i > 1 else 3,
                 learning_rate=.0001*.5**max(i-2, 0)) for i in range(1, 5)]
    return dict(epochs=rows, optimizer_updates=712, actual_backward_calls=1424), split


def test_original_early_stopping_full_coverage_and_two_backward_counts():
    r, split = epoch_fixture(); runner.check_epoch_budget(r, split, 100, 2)


@pytest.mark.parametrize('change', ['partial', 'steps', 'coverage', 'validation', 'backward', 'checkpoint', 'lr'])
def test_rejects_partial_training_or_wrong_full_model_accounting(change):
    r, split = deepcopy(epoch_fixture())
    if change == 'partial': r['epochs'].pop()
    elif change == 'steps': r['epochs'][0]['steps'] = 1
    elif change == 'coverage': r['epochs'][0]['train_windows'] = 32
    elif change == 'validation': r['epochs'][0]['validation_windows'] = 1
    elif change == 'backward': r['actual_backward_calls'] = r['optimizer_updates']
    elif change == 'checkpoint': r['epochs'][1]['checkpoint_saved'] = True
    else: r['epochs'][2]['learning_rate'] = .0001
    with pytest.raises(ValueError): runner.check_epoch_budget(r, split, 100, 2)


def test_real_metadata_entry_remains_cold():
    code = ('import sys;from scripts.flow_matching.register_anomaly_transformer_full_checkpoint_review_v1 import registration;'
        'from scripts.flow_matching.run_anomaly_transformer_full_checkpoint_review_v1 import verify;verify(registration());'
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;'
        'a=install_facades();a["resource_snapshot"]();assert not any(n in sys.modules for n in ["numpy","pandas","torch"])')
    p = subprocess.run([sys.executable, '-B', '-c', code], cwd=runner.ROOT, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr


def test_failed_or_missing_result_is_not_treated_as_pending_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    case = dict(config=dict(artifact_root=str(tmp_path/'models'), data_cache_root=str(tmp_path/'cache'),
                state_root='state'), job=dict(id='fixture', dataset='SMD'))
    assert runner.source_ready(case) is False
    _, _, receipt = runner.source_paths(case)
    runner.write(receipt, dict(job='fixture', passed=False, exit_code=1, resource_guard_aborted=False))
    with pytest.raises(ValueError, match='guard failed'): runner.source_ready(case)
    runner.write(receipt, dict(job='fixture', passed=True, exit_code=0, resource_guard_aborted=False))
    with pytest.raises(ValueError, match='no completed source result'): runner.source_ready(case)
    runner.write(tmp_path/'models/fixture/result.json', dict(status='completed'))
    assert runner.source_ready(case) is True  # Full content verification remains the guarded child's job.


@pytest.mark.parametrize('family', ['main', 'plain', 'max', 'sensitivity'])
def test_fresh_full_width_checkpoint_replay_and_independent_arrays_fixture(tmp_path, family):
    # Real full-width training/replay on a tiny fixture; never benchmark performance.
    code = r'''
import json,pathlib,sys,os,numpy as np,torch
from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent
from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as review
root=pathlib.Path(sys.argv[1]);family=sys.argv[2];data=root/'data/toy';data.mkdir(parents=True)
(root/'configs/datasets').mkdir(parents=True)
rng=np.random.default_rng(1103)
np.save(data/'train.npy',rng.normal(size=(500,3)));np.save(data/'test.npy',rng.normal(size=(105,3)))
y=np.zeros(105,dtype=np.int8);y[20:40]=1;np.save(data/'labels.npy',y)
(root/'configs/datasets/mtsad_smd.json').write_text(json.dumps(dict(loader='npy',directory='toy',files=dict(train='train.npy',test='test.npy',label='labels.npy'))))
contract=dict(reported_table13_splits=dict(SMD=dict(train=400,validation=100,test=105,features=3)),paper_evaluation=dict(r_percent=dict(SMD=.5),fixed_delta=dict(SMD=.1),ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]))
source=parent.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'
config=dict(artifact_root=str(root/'artifacts'),data_cache_root=str(root/'cache'),data_root='data',source_directory=str(source),state_root='state',contract_sha256='fixture',explicit_reconstruction_choices=['fixture only'],boundary='fixture plumbing only')
job=dict(id='fixture',dataset='SMD',seed=1103)
path=root/'config.json';path.write_text(json.dumps(config));parent.ROOT=root
if family=='main':
 worker=lambda:parent.run_worker(config,job,path)
 parent.verify=lambda _:contract
elif family=='plain':
 from scripts.flow_matching import run_plain_transformer_paper_baseline_v1 as s
 worker=lambda:s.bound_worker(verify_function=lambda _:contract,root=root)(config,job,path)
elif family=='max':
 from scripts.flow_matching import run_anomaly_transformer_paper_max_ablation_v1 as s
 job['variant']='fixed_sigma1_prior_max'
 worker=lambda:s.bound_worker(job['variant'],verify_function=lambda _:contract,root=root)(config,job,path)
else:
 from scripts.flow_matching import run_anomaly_transformer_paper_sensitivity_v1 as s
 job['parameters']=dict(window=50,hidden=512,layers=3,weight=3)
 worker=lambda:s.bound_worker(job['parameters'],verify_function=lambda _:contract,root=root)(config,job,path)
worker()
# Save only fixture bindings. Production verifier is tested separately on full counts.
(root/'case.json').write_text(json.dumps(dict(family=family,path=str(path),config=config,job=job)))
(root/'contract.json').write_text(json.dumps(contract))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    train = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code, str(tmp_path), family],
        cwd=runner.ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert train.returncode == 0, train.stdout[-1000:]+train.stderr[-4000:]
    replay = r'''
import pathlib,sys
from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as s
root=pathlib.Path(sys.argv[1]);case=s.read(root/'case.json');contract=s.read(root/'contract.json')
s.ROOT=root
def fixture_check(c,k):
 base,cache,_=s.source_paths(c);r=s.read(base/'result.json')
 return r,s.read(cache/'manifest.json'),dict(result_sha256=s.sha(base/'result.json'))
s.check_source=fixture_check
runtime=dict(artifact_root=str(root/'replays'),_config_path='config.json')
if sys.argv[2]=='native':s.native_worker(runtime,case,contract)
else:s.review_worker(runtime,case,contract)
'''
    for phase in ['native', 'independent']:
        p = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', replay, str(tmp_path), phase],
            cwd=runner.ROOT, env=env, capture_output=True, text=True, timeout=180)
        assert p.returncode == 0, p.stdout[-1000:]+p.stderr[-4000:]
    proof = runner.read(tmp_path/'replays/fixture/independent/review.json')
    assert proof['passed'] and proof['all_original_streams_and_covered_arrays_checked']
    assert not proof['strict_TAB_result'] and not proof['paper_equivalence_certified']
