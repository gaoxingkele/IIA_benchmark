"""All original downstream source splits/horizons;fixtures are not performance."""
import ast
import os
import subprocess
import sys
import pytest
from scripts.flow_matching import prepare_moment_icm_forecast_evaluation_inputs_v1 as s


def test_all6_datasets_all18_horizon_pairs_and_original_data_policy():
    c = s.registration(); jobs = s.verify(c)
    assert len(jobs) == 6 and c['horizons'] == [96, 192, 384] and c['sequence_length'] == 256
    assert {j['dataset'] for j in jobs} == {'Exchange', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather'}
    assert c['required_horizon_dataset_pairs'] == 18 and c['roles'] == ['train', 'val', 'test']
    assert c['other_dataset_ratios'] == dict(train=.7, validation=.1, test=.2)
    parent = s.read(s.ROOT/'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json')
    assert c['settings'] == parent['settings'] and c['resource_lock'] == parent['resource_lock']


@pytest.mark.parametrize('change', ['jobs', 'horizons', 'other_dataset_ratios', 'settings', 'stride'])
def test_reject_dropped_datasets_horizons_pretraining_ratios_and_changed_resources(change):
    c = s.registration()
    if change == 'jobs': c[change].pop()
    elif change == 'horizons': c[change] = [96]
    elif change == 'other_dataset_ratios': c[change] = dict(train=.6, validation=.1, test=.3)
    elif change == 'settings': c[change]['maximum_worker_private_bytes'] += 1
    else: c[change] = 2
    with pytest.raises(ValueError, match='All6datasets'):
        s.verify(c)


@pytest.mark.parametrize('dataset', ['exchange_rate', 'ETTh1'])
def test_actual_source_all_roles_horizons_axes_and_target_boundaries_on_fixture(tmp_path, dataset):
    code = r'''
from pathlib import Path
import sys,numpy as np,pandas as pd
from scripts.flow_matching import prepare_moment_icm_forecast_evaluation_inputs_v1 as s
root=Path(sys.argv[1]);name=sys.argv[2];n=16000
path=root/(name+'.csv')
t=np.arange(n,dtype=float)
pd.DataFrame({'date':np.arange(n),'a':t,'b':2*t+np.sin(t),'OT':3*t-np.cos(t)}).to_csv(path,index=False)
cls=s.original.source_long_class(s.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original')
record=s.export_dataset(cls,path,root/'output')
assert record['full_window_populations_and_horizons_retained'] and not record['forecast_target_roles_overlap']
assert record['total_source_raw_rows']==n and record['training_scaler_fixed_across_all_roles']
for role,row in record['roles'].items():
 saved=np.load(root/'output'/role/'values_TC.npy',allow_pickle=False)
 rows=np.load(root/'output'/role/'original_CSV_rows.npy',allow_pickle=False)
 assert saved.shape[1]==3 and np.array_equal(np.diff(rows),np.ones(len(rows)-1))
 for h in [96,192,384]:
  ds=s.source_dataset(cls,path,role,h)
  assert saved.tobytes()==ds.data.tobytes(order='C')
  r=next(x for x in row['windows_by_horizon'] if x['horizon']==h)
  assert r['windows']==len(ds)==len(saved)-256-h+1
  assert r['target_coordinates']==len(ds)*3*h and r['every_window_bound_checked']
  if role=='val':assert r['source_target_row_start']==ds._get_borders().train.stop
  if role=='test':assert r['source_target_row_start']==ds._get_borders().val.stop
  for i in [0,len(ds)//2,len(ds)-1]:
   item=ds[i]
   assert np.array_equal(item.timeseries,saved[i:i+256].T)
   assert np.array_equal(item.forecast,saved[i+256:i+256+h].T)
if name=='exchange_rate':
 assert record['roles']['train']['time_rows']==11200
 assert record['original_rows_outside_source_fixed_month_selection']==0
else:
 assert record['roles']['train']['time_rows']==8640
 assert record['selected_source_time_end_exclusive']==14400
 assert record['original_rows_outside_source_fixed_month_selection']==1600
assert path.exists() and len(pd.read_csv(path))==n and 'torch' not in sys.modules
'''
    result = subprocess.run([str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8', '-c', code,
        str(tmp_path), dataset], cwd=s.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-1500:]+result.stderr[-4500:]


@pytest.mark.parametrize('role,horizon', [('future',96),('train',12)])
def test_unknown_split_or_forecast_horizon_is_not_silently_substituted(role,horizon):
    with pytest.raises(ValueError): s.source_dataset(None, 'not_opened', role, horizon)


def test_cold_registration_does_not_load_numeric_packages():
    code = '''
import sys
from scripts.flow_matching import prepare_moment_icm_forecast_evaluation_inputs_v1 as s
c=s.registration();s.verify(c)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout[-1000:]+result.stderr[-2000:]


def test_no_guard_or_resource_adaptation_in_private_original_controller():
    # Only the two descriptive state strings may differ from the pinned source.
    from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as old
    from types import CodeType
    replacements = {'closing_all1949_anomaly_sources': 'preparing_all6_forecast_downstream_sources',
                    'all1949_anomaly_cache_closure_passed': 'all6_forecast_inputs_all18_horizons_prepared'}
    def semantics(code, rename=False):
        constants = []
        for item in code.co_consts:
            if isinstance(item, CodeType): constants.append(semantics(item, rename))
            elif rename and isinstance(item,str): constants.append(replacements.get(item,item))
            else: constants.append(item)
        return code.co_code, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, tuple(constants)
    actual = s.guarded_controller()
    # Compare the unchanged source function in the same isolated compilation
    # context. Python3.12 may encode calls on imported module globals using
    # different equivalent LOAD_GLOBAL/LOAD_ATTR NULL flags in a full module.
    node = next(n for n in ast.parse(s.Path(old.__file__).read_text(encoding='utf-8')).body
                if isinstance(n,ast.FunctionDef) and n.name=='controller')
    pristine = {}
    exec(compile(ast.Module(body=[node],type_ignores=[]), str(old.__file__), 'exec'), pristine)
    assert semantics(actual.__code__) == semantics(pristine['controller'].__code__, True)
    assert actual.__globals__['MODULE'] == s.MODULE and actual.__globals__['CONFIG'] == s.CONFIG
    assert actual.__globals__['REPORT'] == s.REPORT
