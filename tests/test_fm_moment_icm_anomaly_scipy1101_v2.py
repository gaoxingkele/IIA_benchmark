import os
from pathlib import Path
import subprocess
import pytest
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as r


@pytest.mark.parametrize('slot',[0,160,321])
def test_real_fresh_private_environment_reveals_nonfinite_source_and_keeps_export_guard(tmp_path,slot):
    code=r'''
import sys
from pathlib import Path
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as r
from scripts.flow_matching import prepare_moment_icm_PILE_anomaly_v1 as p
config=p.read(p.ROOT/p.CONFIG);jobs=p.verify(config)
ok,failed,pins=r.source_partition(config,jobs)
assert len(ok)==1627 and len(failed)==322
job=failed[int(sys.argv[1])]
r.install_private_path({})
import numpy as np,scipy,importlib.metadata as m
assert scipy.__version__=='1.10.1' and m.version('pandas')=='2.2.2' and np.__version__=='1.25.2'
assert str(Path(scipy.__file__).resolve()).startswith(str(Path(p.read(p.ROOT/r.PACKAGE)['private_target']).resolve()))
cls=p.source_anomaly_class(p.ROOT/config['source_root']);raw,rows=p.csv_identity(job['raw'])
nonfinite=0
for split in p.SPLITS:
 data=p.source_dataset(cls,p.ROOT/job['raw']['path'],split)
 out=Path(sys.argv[2])/split
 if not np.isfinite(data.data).all():
  nonfinite+=int((~np.isfinite(data.data)).sum())
  try:p.export_split(data,job,rows,out)
  except ValueError as error:assert 'finite' in str(error)
  else:raise AssertionError('Nonfinite source was silently accepted')
 else:
  record=p.export_split(data,job,rows,out)
  assert np.load(out/'values_T.npy').tobytes()==data.data.tobytes()
  assert record['all_source_signal_coordinates_byte_equal']
  assert record['source_scaler_fit_rows']==data.scaler.n_samples_seen_
assert nonfinite>0
assert p.sha(p.ROOT/job['raw']['path'])==job['raw']['sha256']
assert 'torch' not in sys.modules
'''
    result=subprocess.run([str(r.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8',
        '-c',code,str(slot),str(tmp_path)],cwd=r.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]


def test_private_environment_never_hot_swaps_an_already_imported_scipy(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'scipy',object())
    with pytest.raises(ValueError,match='Fresh private repair worker'):r.install_private_path({})
