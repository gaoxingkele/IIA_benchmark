import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as p


def test_only_nonfinite_inputs_are_replaced_and_imputed_targets_never_become_observed():
    original=np.array([-0.,3.,np.nan,np.inf,7.,-np.inf],dtype=np.float64)
    observed=np.array([1,0,0,1,1,0],dtype=bool)
    values,targets,filled=p.restored_vectors(original,observed,8.,2.)
    assert values.tobytes()==np.array([-0.,3.,0.,0.,7.,0.],dtype=np.float64).tobytes()
    assert np.array_equal(targets,[1,0,0,0,1,0])
    assert np.array_equal(filled,[0,0,1,1,0,1])
    assert np.isnan(original[2]) and np.isposinf(original[3])


@pytest.mark.parametrize('mean,scale',[(np.nan,1.),(0.,np.nan),(0.,0.)])
def test_no_all_missing_training_scaler_or_invalid_scale_is_silently_invented(mean,scale):
    with pytest.raises(ValueError):p.restored_vectors(np.array([np.nan]),np.array([False]),mean,scale)


def test_loss_uses_mask_zero_predict_convention_and_excludes_raw_missing_targets():
    valid=np.array([[[True,False,True,True],[False,True,True,True]]])
    keep=np.array([[0,0,1,0]]);padding=np.array([[1,1,1,0]])
    actual=p.reconstruction_loss_mask(keep,valid,padding)
    assert np.array_equal(actual,np.array([[[True,False,False,False],[False,True,False,False]]]))
    with pytest.raises(ValueError):p.reconstruction_loss_mask([[2]],valid,padding)


@pytest.mark.parametrize('slot',[0,160,321])
def test_real_source_all_splits_preserve_finite_values_and_complete_raw_target_identity(tmp_path,slot):
    code=r'''
import sys
from pathlib import Path
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as p
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as r
original=r.read(r.ROOT/r.producer.CONFIG);all_jobs=r.producer.verify(original)
ok,jobs,pins=r.source_partition(original,all_jobs);job=jobs[int(sys.argv[1])]
r.install_private_path({})
import numpy as np,pandas as pd
raw,rows=r.producer.csv_identity(job['raw']);observed=np.isfinite(pd.read_csv(r.ROOT/job['raw']['path']).iloc[:,0].to_numpy())
cls=r.producer.source_anomaly_class(r.ROOT/original['source_root']);replacements=0
for split in r.producer.SPLITS:
 data=r.producer.source_dataset(cls,r.ROOT/job['raw']['path'],split)
 out=Path(sys.argv[2])/split;record=p.export_split(data,job,rows,observed,out)
 replacements+=record['replaced_residual_nonfinite_coordinates']
 source=np.load(out/p.ARRAYS['source_values']);values=np.load(out/p.ARRAYS['values']);valid=np.load(out/p.ARRAYS['valid_targets'])
 assert source.tobytes()==data.data.tobytes()
 finite=np.isfinite(source);assert values[finite].tobytes()==source[finite].tobytes()
 assert np.isfinite(values).all() and np.all(values[~finite]==0.)
 ids=np.load(out/p.ARRAYS['raw_rows']);positions=np.searchsorted(rows,ids)
 assert np.array_equal(valid,observed[positions]&finite)
 assert len(ids)==len(values)==len(data.data) and record['raw_time_rows_dropped']==0
 if len(values)>=256:
  for index in sorted({0,(len(values)-256)//2,len(values)-256}):
   item=p.read_window(out,index)
   assert item['timeseries'].tobytes()==values[index:index+256].reshape(1,256).tobytes()
   assert np.array_equal(item['valid_targets'],valid[index:index+256].reshape(1,256))
   loss=p.reconstruction_loss_mask(np.zeros((1,256)),item['valid_targets'],item['input_mask'])
   assert np.array_equal(loss,item['valid_targets'])
try:p.export_split(data,job,rows,observed,out)
except FileExistsError:pass
else:raise AssertionError('Existing cache overwritten')
assert replacements>0 and r.sha(r.ROOT/job['raw']['path'])==job['raw']['sha256']
assert 'torch' not in sys.modules
'''
    result=subprocess.run([str(p.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8',
        '-c',code,str(slot),str(tmp_path)],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]
