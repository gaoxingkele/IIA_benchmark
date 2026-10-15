"""Keep all source instances,exclude TEST fitting and disclose grid differences."""
from copy import deepcopy
import os
import subprocess
import pytest
from scripts.flow_matching import prepare_moment_icm_classification_train_only_v1 as s


def test_all117_source_successes_and_original_data_worker_policy_remain_bound():
    config = s.registration(); assert len(s.verify(config)) == 117
    parent = s.read(s.ROOT/s.PARENT)
    assert len(config['full_original_source_preprocessing_bindings']) == 117
    assert config['settings'] == parent['settings'] and config['resource_lock'] == parent['resource_lock']
    assert config['jobs'] == parent['jobs']


@pytest.mark.parametrize('field', ['jobs', 'policy', 'settings', 'full_original_source_preprocessing_bindings'])
def test_reject_partial_scope_test_fit_relaxed_caps_or_changed_source_binding(field):
    config = s.registration()
    if field == 'jobs': config[field].pop()
    elif field == 'policy': config[field]['TEST_statistic_fit'] = True
    elif field == 'settings': config[field]['maximum_worker_private_bytes'] += 1
    else: config[field][0]['proof_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Full117'):
        s.verify(config)


@pytest.mark.parametrize('scenario', ['equal', 'different_length', 'nan', 'unscaled'])
def test_actual_source_train_scaler_is_fixed_and_all_instance_coordinates_retained(tmp_path, scenario):
    code = r'''
import pathlib,sys,numpy as np
from scripts.flow_matching import prepare_moment_icm_classification_train_only_v1 as s
root=pathlib.Path(sys.argv[1]);scenario=sys.argv[2]
name='Chinatown' if scenario=='unscaled' else 'fixture'
d=root/name;d.mkdir()
header='@problemName fixture\n@timestamps false\n@univariate false\n@classLabel true A B\n@data\n'
def raw(offset,length):
 rows=[]
 for i in range(4):
  signals=[]
  for c in range(2):
   signals.append(','.join('?' if scenario=='nan' and i==1 and k==3 else str(offset+i*(c+1)+k) for k in range(length)))
  rows.append(':'.join(signals)+':'+('A' if i%2==0 else 'B')+'\n')
 return header+''.join(rows)
train=d/(name+'_TRAIN.ts');test=d/(name+'_TEST.ts')
train.write_text(raw(0,9));test.write_text(raw(100,13 if scenario=='different_length' else 9))
cls,small,unscaled=s.source.source_components(s.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original')
a=s.source.source_dataset(cls,train,name,small,'train')
source=s.source.source_dataset(cls,test,name,small,'test')
fixed,proxy=s.heldout_class(cls,a.scaler)
b=s.source.source_dataset(fixed,test,name,small,'test')
assert proxy.test_fit_calls==0 and len(a)==len(b)==8
s.source.export_split(a,4,root/'sourceTRAIN')
checked=s.verify_training_coordinates(a,root/'sourceTRAIN',4)
assert checked['checked_value_coordinates']==4*2*256
s.source.export_split(source,4,root/'sourceTEST')
s.source.export_split(b,4,root/'strictTEST')
comparison=s.compare_complete_heldout(root/'strictTEST',root/'sourceTEST')
assert comparison['labels.npy']['changed_coordinates']==0
assert comparison['original_instance_rows.npy']['changed_coordinates']==0
if scenario=='unscaled':
 assert proxy.transform_calls==0 and all(r['changed_coordinates']==0 for r in comparison.values())
else:
 assert proxy.transform_calls==1 and comparison['values.npy']['changed_coordinates']>0
 assert np.array_equal(proxy.mean_,a.scaler.mean_) and np.array_equal(proxy.scale_,a.scaler.scale_)
 if scenario=='different_length':assert proxy.resampled_test_grid==dict(original_test_native_length=13,training_native_length=9)
 else:assert proxy.resampled_test_grid is None
try:proxy.fit(np.zeros((2,9)))
except ValueError:pass
else:raise AssertionError('Heldout fit must always fail')
assert 'torch' not in sys.modules
'''
    result = subprocess.run([str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8', '-c', code,
                             str(tmp_path), scenario], cwd=s.ROOT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-1500:]+result.stderr[-4000:]


def test_controller_uses_original_guard_body_and_only_new_module_binding(monkeypatch):
    captured = {}
    original = s.types.FunctionType
    def capture(code, namespace, *args):
        captured.update(code=code, namespace=namespace)
        return lambda config, jobs: None
    monkeypatch.setattr(s.types, 'FunctionType', capture)
    s.controller({}, [])
    assert captured['code'] is s.source.controller.__code__
    assert captured['namespace']['MODULE'] == s.MODULE
    assert captured['namespace']['sha'] is s.source.sha
