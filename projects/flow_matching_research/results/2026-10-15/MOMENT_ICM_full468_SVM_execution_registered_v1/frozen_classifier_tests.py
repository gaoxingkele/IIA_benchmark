"""Full classifier scope and real source primitives;fixtures are not performance."""
import ast
from copy import deepcopy
import json
import math
import os
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import run_moment_icm_full_classification_v1 as s


def test_full468_cases_cover_every_dataset_model_and_preprocessing_track_once():
    config = s.registration(); jobs = s.verify(config)
    assert len(jobs) == 468 and len({j['dataset'] for j in jobs.values()}) == 117
    assert {j['variant'] for j in jobs.values()} == {'independent', 'icm'}
    assert {j['track'] for j in jobs.values()} == {'source_TEST_fitted', 'TRAIN_fitted_control'}
    assert all(j['checkpoint'] == 'full_final_post_scheduler_39479' and j['seed'] == 13 for j in jobs.values())
    for variant in ['independent', 'icm']:
        for track in ['source_TEST_fitted', 'TRAIN_fitted_control']:
            assert len([j for j in jobs.values() if j['variant'] == variant and j['track'] == track]) == 117
    policy = s.read(s.ROOT/s.POLICY)
    assert config['settings'] == policy['settings'] and config['resource_lock'] == policy['resource_lock']
    assert config['original_paper_scalar_targets'] == 702


@pytest.mark.parametrize('change', ['jobs', 'pooling_tolerance', 'resource_lock', 'settings', 'SVM_source_sha256'])
def test_reduced_scope_loose_math_or_changed_resources_rejected(change):
    config = s.registration()
    if change == 'jobs': config[change].pop()
    elif change == 'pooling_tolerance': config[change]['atol'] = .01
    elif change == 'settings': config[change]['minimum_commit_headroom_bytes'] -= 1
    else: config[change] = 'changed'
    with pytest.raises(ValueError, match='All468'):
        s.verify(config)


def test_no_numerical_imports_or_random_release_weights_when_training_is_missing(monkeypatch, tmp_path):
    # Complete public release checkpoints already on disk must never satisfy
    # the private full-PILE model requirement.
    monkeypatch.setattr(s.science, 'trained_checkpoint', lambda config, variant: None)
    monkeypatch.setattr(s.science, 'dataset_binding', lambda *args: {})
    with pytest.raises(ValueError, match='before numerical imports'):
        s.science.run_case({}, dict(variant='icm', dataset='fixture', track='source_TEST_fitted'), tmp_path/'never_created')
    assert not (tmp_path/'never_created').exists()


@pytest.mark.parametrize('case', ['balanced', 'single_class', 'unseen_prediction'])
def test_complete_scalar_confusion_and_macro_f1_match_independent_sklearn(case):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    y = np.array([0, 1, 2, 0, 1, 2]); p = np.array([0, 2, 2, 1, 1, 2])
    if case == 'single_class': y = p = np.zeros(7, dtype=int)
    if case == 'unseen_prediction': p[-1] = 3
    m = s.science.independent_classification(y, p)
    assert m['accuracy'] == accuracy_score(y, p)
    assert math.isclose(m['macro_F1'], f1_score(y, p, average='macro'), abs_tol=1e-12)
    assert m['confusion'] == confusion_matrix(y, p, labels=m['classes']).tolist()
    assert m['descriptive_Wilson95_low'] <= m['accuracy'] <= m['descriptive_Wilson95_high']
    assert not m['model_seed_uncertainty_estimated'] and not m['unknown_subject_group_dependence_accounted_for']


@pytest.mark.parametrize('problem', ['short_prediction', 'empty', 'two_dimensional'])
def test_partial_classifier_predictions_cannot_produce_a_result(problem):
    y = np.array([0, 1]); p = np.array([0, 1])
    if problem == 'short_prediction': p = p[:1]
    elif problem == 'empty': y = p = np.array([])
    else: y = y[:, None]
    with pytest.raises(ValueError): s.science.independent_classification(y, p)


def test_real_pinned_source_pooling_and_SVM_paths_in_actual_producer_environment(tmp_path):
    # Actual original sklearn/Torch with mathematical fixture features only;
    # the 10-worker CV object is observed with a fake search to avoid launching
    # a scientific resource-consuming grid outside the guarded job queue.
    code = r'''
import numpy as np,torch,warnings,json
from sklearn.svm import SVC
from sklearn.exceptions import FitFailedWarning
import sklearn.model_selection as selection
from scripts.flow_matching import moment_icm_svm_classification_v1 as s
torch.set_num_threads(2)
class Core:
 def __call__(self,x,input_mask,reconstruction_mask):
  b,c,t=x.shape
  return torch.arange(32,dtype=torch.float32)[None,None,:,None].expand(b,c,32,256)+torch.arange(c,dtype=torch.float32)[None,:,None,None]
x=np.zeros((1,3,256));m=np.ones((1,256),dtype=np.int64);m[:,:8]=0
f,e=s.pooled_features(Core(),x,m,dict(rtol=2e-6,atol=2e-6))
assert f.shape==(1,3*256) and np.allclose(f.reshape(3,256)[:,0],[16,17,18]) and e==0
for bad in [np.full((1,256),.5),np.pad(np.ones((1,7)),((0,0),(249,0)))]:
 try:s.pooled_features(Core(),x,bad,dict(rtol=2e-6,atol=2e-6))
 except ValueError:pass
 else:raise AssertionError('Invalid or empty full patch mask accepted')
features=np.random.RandomState(13).normal(size=(20,4));labels=np.arange(20)%2
estimator,audit=s.source_svm(features,labels,s.sha(s.ROOT/s.SVM_SOURCE))
assert estimator.C==100000 and audit['grid']==[] and not audit['test_labels_used_for_selection']
hard=SVC(C=np.inf).fit(features,labels)
assert np.isinf(hard.C) and np.isfinite(hard.dual_coef_).all()
seen={}
class Search:
 best_infinite=False
 def __init__(self,svm,grid,**kwargs):
  assert kwargs==dict(cv=5,n_jobs=10) and len(grid['C'])==10 and np.isinf(grid['C'][-1])
  self.grid=grid;seen.update(grid=grid,kwargs=kwargs)
 def fit(self,x,y):
  self.best_estimator_=SVC(C=np.inf if self.best_infinite else 1).fit(x,y)
  self.cv_results_=dict(params=[dict(C=c) for c in self.grid['C']],mean_test_score=np.array([.5]*9+[.9 if self.best_infinite else np.nan]),rank_test_score=np.arange(1,11))
  if not self.best_infinite:warnings.warn('retained source failed-candidate fixture',FitFailedWarning)
selection.GridSearchCV=Search
features=np.random.RandomState(13).normal(size=(60,4));labels=np.arange(60)%2
estimator,audit=s.source_svm(features,labels,s.sha(s.ROOT/s.SVM_SOURCE))
assert len(audit['grid'])==10 and audit['grid'][-1]['candidate_failed'] and audit['grid'][-1]['C']=='positive_infinity'
assert audit['grid'][-1]['mean_training_CV_accuracy'] is None and audit['warnings'][0]['category']=='FitFailedWarning'
assert not audit['test_labels_used_for_selection']
Search.best_infinite=True
estimator,audit=s.source_svm(features,labels,s.sha(s.ROOT/s.SVM_SOURCE))
assert np.isinf(estimator.C) and audit['chosen_parameters']['C']=='positive_infinity'
assert audit['actual_chosen_C_is_positive_infinity'] and not audit['grid'][-1]['candidate_failed']
json.dumps(audit,allow_nan=False)
'''
    result = subprocess.run([str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8', '-c', code],
        cwd=s.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-1500:]+result.stderr[-5000:]


def test_source_periodic_callback_is_AD_and_not_silently_marked_as_classification():
    path = s.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original/scripts/evaluation/evaluation.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    commands = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'COMMANDS' for t in n.targets))
    assert len(commands) == 1 and commands[0][1].endswith('zero_shot/anomaly_detection.py')


def test_registration_and_missing_training_gate_remain_numerically_cold():
    code = '''
import json,sys
from scripts.flow_matching import run_moment_icm_full_classification_v1 as s
c=s.registration();s.verify(c)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
print(json.dumps(dict(passed=True,cases=len(c['jobs']))))
'''
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-1000:]+result.stderr[-2500:]
    assert json.loads(result.stdout)['cases'] == 468


def test_old_success_cannot_bypass_missing_current_scientific_prerequisites(monkeypatch, tmp_path):
    monkeypatch.setattr(s.science, 'trained_checkpoint', lambda *a: None)
    monkeypatch.setattr(s.science, 'dataset_binding', lambda *a: {})
    with pytest.raises(ValueError, match='still required'):
        s.science.verify_existing_evidence({}, dict(variant='icm', dataset='fixture', track='source_TEST_fitted'), tmp_path, 'native')


def test_current_all_array_hashes_are_required_in_existing_evidence(tmp_path):
    path = tmp_path/'values.npy'; path.write_bytes(b'changed')
    binding = dict(directories=dict(TRAIN=str(tmp_path), TEST=str(tmp_path)),
                   records=dict(TRAIN=dict(files_sha256={'values.npy': '0'*64}), TEST=dict(files_sha256={})))
    with pytest.raises(ValueError, match='arrays changed'):
        s.science.verify_bound_arrays(binding)
